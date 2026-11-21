"""Загрузка и объединение дополнительных наборов данных с HuggingFace.

Модуль объединяет базовый корпус UCI ``SMSSpamCollection`` (через
:func:`ml.data.load_sms.load_sms_dataset`) с внешними источниками
HuggingFace для расширения русскоязычного и казахскоязычного покрытия:

1. ``ruSpamModels/russian-spam-detection`` (``russian_spam_detection.parquet``) —
   русскоязычные сообщения с разметкой ``0/1`` (могут встречаться казахские строки).
2. ``AugustLight/telegram_spam_ru`` (``telegram_spam_ru.jsonl``) —
   русскоязычный Telegram-спам с разметкой ``0/1``.
3. ``issai/kazsandra`` и ``issai/kazparc`` (``issai_kazsandra.csv``,
   ``issai_kazparc.csv``) — неразмеченные по спаму казахские тексты
   (используются как безопасные ``label=0`` при строгом квотировании).
4. ``kz-transformers/multidomain-kazakh-dataset``
   (``kz_transformers_leipzig_sample.csv`` / ``kz_transformers_leipzig.csv``) —
   казахский корпус Leipzig (используется как ``label=0`` при строгом квотировании).
5. ``salyamq/kazspam-v1`` (``kazspam_v1.jsonl``) — размеченные казахские
   спам-тексты (берутся короткие казахские ``casino`` и ограниченная квота
   ``porn`` как ``label=1``).

Итоговый :class:`pandas.DataFrame` приводится к единой схеме столбцов:
``["id", "text", "label", "source", "language"]``.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from ml.data.load_sms import load_sms_dataset

logger = logging.getLogger(__name__)

# Корень проекта /home/kovrik/sms_guard (ml/data/load_hf_datasets.py -> parents[2]).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
HF_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw" / "hf"

RUSPAM_FILENAME: str = "russian_spam_detection.parquet"
TELEGRAM_SPAM_FILENAME: str = "telegram_spam_ru.jsonl"
ISSAI_KAZSANDRA_FILENAME: str = "issai_kazsandra.csv"
ISSAI_KAZPARC_FILENAME: str = "issai_kazparc.csv"
KZ_TRANSFORMERS_SAMPLE_FILENAME: str = "kz_transformers_leipzig_sample.csv"
KZ_TRANSFORMERS_FILENAME: str = "kz_transformers_leipzig.csv"
KAZSPAM_V1_FILENAME: str = "kazspam_v1.jsonl"

STANDARD_COLUMNS: tuple[str, ...] = ("text", "label", "source", "language")
COMBINED_COLUMNS: tuple[str, ...] = ("id", "text", "label", "source", "language")

_KAZAKH_CHAR_RE = re.compile(r"[әғқңөұүһӘҒҚҢӨҰҮҺ]")
_EXCLUDED_WEB_RE = re.compile(r"(?:http|wikipedia)", re.IGNORECASE)
_DEFAULT_CHUNK_SIZE: int = 1024 * 1024
_DEFAULT_SAMPLE_BYTES: int = 15_000_000


def _empty_hf_frame() -> pd.DataFrame:
    """Вернуть пустой ``DataFrame`` со стандартными столбцами HF-источников."""
    return pd.DataFrame(
        {
            "text": pd.Series(dtype="object"),
            "label": pd.Series(dtype="int64"),
            "source": pd.Series(dtype="object"),
            "language": pd.Series(dtype="object"),
        },
        columns=list(STANDARD_COLUMNS),
    )


def _finalize_hf_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Привести столбцы и типы ``DataFrame`` к единому формату источников."""
    if df.empty:
        return _empty_hf_frame()

    out = df.loc[:, list(STANDARD_COLUMNS)].copy().reset_index(drop=True)
    out["text"] = out["text"].astype("object")
    out["label"] = out["label"].astype("int64")
    out["source"] = out["source"].astype("object")
    out["language"] = out["language"].astype("object")
    return out


def _sample_rows(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    """Детерминированно ограничить число строк в ``DataFrame``."""
    if max_rows <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state).copy()


def _normalize_label_series(series: pd.Series) -> pd.Series:
    """Преобразовать столбец меток в числовой формат (``0``, ``1`` или ``NaN``)."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.where(numeric.isin([0, 1]))

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"0": 0, "1": 1, "ham": 0, "spam": 1, "false": 0, "true": 1})
    )
    return pd.to_numeric(mapped, errors="coerce")


def _filter_clean_texts(
    texts: pd.Series,
    min_len: int,
    max_len: int,
    drop_web: bool = False,
) -> pd.Series:
    """Вернуть очищенную серию строк в заданных границах длины."""
    cleaned = texts.dropna().astype(str).str.strip()
    mask = cleaned.str.len().between(min_len, max_len)
    if drop_web:
        mask &= ~cleaned.str.contains(_EXCLUDED_WEB_RE, regex=True, na=False)
    return cleaned.loc[mask]


def get_hf_token() -> str | None:
    """Получить токен HuggingFace из окружения или локального кэша.

    Последовательно проверяет переменные окружения ``HF_TOKEN``,
    ``HUGGING_FACE_HUB_TOKEN``, а затем файл ``~/.cache/huggingface/token``.
    Никогда не записывает значение токена в логи.
    """
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        try:
            cached = token_file.read_text(encoding="utf-8").strip()
            if cached:
                return cached
        except OSError:
            return None
    return None


def download_hf_file(
    repo_id: str,
    remote_path: str,
    dest_path: Path,
    max_bytes: int | None = None,
    timeout: int = 120,
) -> tuple[bool, str]:
    """Скачать файл датасета с HuggingFace Hub без утечки токена и битых файлов.

    Parameters
    ----------
    repo_id:
        Идентификатор репозитория датасета, например ``"issai/kazsandra"``.
    remote_path:
        Путь к файлу внутри репозитория, например ``"processed_combined.parquet"``.
    dest_path:
        Локальный путь сохранения файла.
    max_bytes:
        Необязательный префиксный лимит скачивания в байтах (через HTTP ``Range``).
        Для ``.csv``-файлов неполная последняя строка обрезается по последнему
        переводу строки ``\\n``.
    timeout:
        Таймаут HTTP-запроса в секундах.

    Returns
    -------
    tuple[bool, str]
        ``(True, "cached")`` если непустой файл уже существует, ``(True, "downloaded")``
        при успешной загрузке, либо ``(False, "<причина>")`` при ошибке
        (включая HTTP 401/403 для закрытых наборов без падения по исключению).
    """
    target = Path(dest_path)
    if target.exists() and target.stat().st_size > 0:
        return True, "cached"

    clean_remote = remote_path.lstrip("/")
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{clean_remote}"

    headers: dict[str, str] = {"User-Agent": "sms-guard-hf-loader/1.0"}
    token = get_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if max_bytes is not None and max_bytes > 0:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"

    request = urllib.request.Request(url, headers=headers)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            chunks: list[bytes] = []
            remaining = max_bytes if (max_bytes is not None and max_bytes > 0) else None
            while True:
                to_read = (
                    min(_DEFAULT_CHUNK_SIZE, remaining)
                    if remaining is not None
                    else _DEFAULT_CHUNK_SIZE
                )
                if to_read <= 0:
                    break
                chunk = response.read(to_read)
                if not chunk:
                    break
                chunks.append(chunk)
                if remaining is not None:
                    remaining -= len(chunk)

        payload = b"".join(chunks)
        if not payload:
            return False, "empty response"

        if max_bytes is not None and target.suffix.lower() == ".csv":
            last_newline = payload.rfind(b"\n")
            if last_newline > 0:
                payload = payload[: last_newline + 1]

        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".part",
            delete=False,
        ) as tmp_fh:
            tmp_path = Path(tmp_fh.name)
            tmp_fh.write(payload)

        tmp_path.replace(target)
        tmp_path = None
        return True, "downloaded"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            reason = exc.reason or "access denied"
            logger.warning("Нет доступа к %s/%s (HTTP %s)", repo_id, clean_remote, exc.code)
            return False, f"401/403: HTTP {exc.code} {reason}"
        reason = exc.reason or "http error"
        logger.warning("Ошибка скачивания %s/%s: HTTP %s", repo_id, clean_remote, exc.code)
        return False, f"HTTP {exc.code}: {reason}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Сетевая ошибка при скачивании %s/%s: %s", repo_id, clean_remote, exc)
        return False, f"download error: {exc}"
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _sample_ruspam_class(
    df_class: pd.DataFrame,
    limit: int,
    include_kazakh_rows: bool,
    random_state: int,
) -> pd.DataFrame:
    """Выбрать до ``limit`` строк одного класса с приоритетом казахских строк."""
    if limit <= 0 or df_class.empty:
        return df_class.iloc[0:0].copy()

    if not include_kazakh_rows:
        ru_only = df_class.loc[df_class["language"] == "ru"]
        return _sample_rows(ru_only, limit, random_state=random_state)

    kk_rows = df_class.loc[df_class["language"] == "kk"]
    ru_rows = df_class.loc[df_class["language"] != "kk"]

    # Сначала сохраняем редкие казахские строки, затем добираем русские до квоты.
    chosen_kk = _sample_rows(kk_rows, limit, random_state=random_state)
    remaining = max(0, limit - len(chosen_kk))
    chosen_ru = _sample_rows(ru_rows, remaining, random_state=random_state)
    return pd.concat([chosen_kk, chosen_ru], ignore_index=True)


def load_ruspam_dataset(
    path: Path | str | None = None,
    max_ham: int = 15000,
    max_spam: int = 10000,
    include_kazakh_rows: bool = True,
    min_len: int = 10,
    max_len: int = 400,
    random_state: int = 42,
) -> pd.DataFrame:
    """Загрузить и сбалансировать набор ``ruSpamModels/russian-spam-detection``.

    Читает Parquet-файл со столбцами ``message`` (или ``text``) и ``label``,
    фильтрует сообщения по длине, удаляет дубликаты, определяет вкрапления
    казахского языка по специфическим казахским буквам и балансирует классы
    согласно ``max_ham`` и ``max_spam``.
    """
    resolved = Path(path) if path is not None else HF_RAW_DIR / RUSPAM_FILENAME
    if not resolved.is_file() or resolved.stat().st_size == 0:
        return _empty_hf_frame()

    try:
        raw = pd.read_parquet(resolved)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не удалось прочитать Parquet %s: %s", resolved, exc)
        return _empty_hf_frame()

    text_col = "message" if "message" in raw.columns else "text"
    if text_col not in raw.columns or "label" not in raw.columns:
        return _empty_hf_frame()

    cleaned_texts = _filter_clean_texts(raw[text_col], min_len=min_len, max_len=max_len)
    if cleaned_texts.empty:
        return _empty_hf_frame()

    labels = _normalize_label_series(raw.loc[cleaned_texts.index, "label"]).dropna()
    if labels.empty:
        return _empty_hf_frame()

    work = pd.DataFrame(
        {
            "text": cleaned_texts.loc[labels.index],
            "label": labels.astype("int64"),
        }
    ).drop_duplicates(subset=["text"])

    is_kk = work["text"].str.contains(_KAZAKH_CHAR_RE, regex=True, na=False)
    work["language"] = "ru"
    work.loc[is_kk, "language"] = "kk"
    work["source"] = "ruSpamModels"

    ham_df = _sample_ruspam_class(
        work.loc[work["label"] == 0],
        limit=max_ham,
        include_kazakh_rows=include_kazakh_rows,
        random_state=random_state,
    )
    spam_df = _sample_ruspam_class(
        work.loc[work["label"] == 1],
        limit=max_spam,
        include_kazakh_rows=include_kazakh_rows,
        random_state=random_state,
    )

    combined = pd.concat([ham_df, spam_df], ignore_index=True)
    return _finalize_hf_frame(combined)


def load_telegram_spam_ru(
    path: Path | str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Загрузить русскоязычный набор ``AugustLight/telegram_spam_ru`` из JSONL."""
    del random_state  # Принимается для единообразия интерфейса загрузчиков.
    resolved = Path(path) if path is not None else HF_RAW_DIR / TELEGRAM_SPAM_FILENAME
    if not resolved.is_file() or resolved.stat().st_size == 0:
        return _empty_hf_frame()

    records: list[dict[str, Any]] = []
    try:
        with resolved.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                raw_text = item.get("text", item.get("message"))
                if raw_text is None:
                    continue
                text = str(raw_text).strip()
                if not text:
                    continue
                raw_label = item.get("label")
                if raw_label is None:
                    continue
                records.append({"text": text, "raw_label": raw_label})
    except OSError as exc:
        logger.warning("Не удалось прочитать JSONL %s: %s", resolved, exc)
        return _empty_hf_frame()

    if not records:
        return _empty_hf_frame()

    df = pd.DataFrame(records)
    df["label"] = _normalize_label_series(df["raw_label"])
    df = df.dropna(subset=["label"]).drop_duplicates(subset=["text"])
    if df.empty:
        return _empty_hf_frame()

    df["label"] = df["label"].astype("int64")
    df["source"] = "telegram_spam_ru"
    df["language"] = "ru"
    return _finalize_hf_frame(df)


def load_kazspam_v1(
    path: Path | str | None = None,
    max_casino: int = 400,
    max_porn: int = 150,
    min_len: int = 15,
    max_casino_len: int = 800,
    max_porn_len: int = 600,
    random_state: int = 42,
) -> pd.DataFrame:
    """Загрузить короткие казахские спам-тексты из ``salyamq/kazspam-v1``.

    Отбираются только записи с казахскими символами из категорий
    ``casino`` (длиной до ``max_casino_len``) и ``porn`` (длиной до
    ``max_porn_len``), чтобы избежать смещения модели на длинные веб-страницы.
    """
    resolved = Path(path) if path is not None else (HF_RAW_DIR / KAZSPAM_V1_FILENAME)
    if not resolved.is_file() or resolved.stat().st_size == 0:
        return _empty_hf_frame()

    limits = {"casino": (max_casino_len, max_casino), "porn": (max_porn_len, max_porn)}
    buckets: dict[str, list[str]] = {"casino": [], "porn": []}

    try:
        with resolved.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                clf = str(item.get("classification", "")).strip().lower()
                if clf not in limits:
                    continue
                max_len, _ = limits[clf]
                text = str(item.get("text", "")).strip()
                if min_len <= len(text) <= max_len and _KAZAKH_CHAR_RE.search(text):
                    buckets[clf].append(text)
    except OSError as exc:
        logger.warning("Не удалось прочитать kazspam-v1 JSONL %s: %s", resolved, exc)
        return _empty_hf_frame()

    parts: list[pd.DataFrame] = []
    for clf, (_, quota) in limits.items():
        texts = buckets[clf]
        if not texts or quota <= 0:
            continue
        sub = pd.DataFrame(
            {
                "text": texts,
                "label": 1,
                "source": "kazspam_v1",
                "language": "kk",
            }
        ).drop_duplicates(subset=["text"])
        parts.append(_sample_rows(sub, max_rows=quota, random_state=random_state))

    if not parts:
        return _empty_hf_frame()

    combined = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["text"])
    return _finalize_hf_frame(combined)


def _extract_kk_from_pair(value: Any) -> str | None:
    """Извлечь казахскую строку из колонки ``pair`` датасета ``issai/kazparc``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, dict):
        kk_val = value.get("kk")
        return str(kk_val) if kk_val is not None else None

    raw = str(value).strip()
    if not raw:
        return None

    if raw.startswith("{") and raw.endswith("}"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
                if isinstance(parsed, dict) and parsed.get("kk") is not None:
                    return str(parsed["kk"])
            except Exception:  # noqa: BLE001
                continue

    if "\t" in raw:
        parts = [part.strip() for part in raw.split("\t") if part.strip()]
        for part in parts:
            if _KAZAKH_CHAR_RE.search(part):
                return part
        if parts:
            return parts[0]

    return raw


def _read_issai_csv(file_path: Path) -> pd.Series:
    """Извлечь серию сырых казахских текстов из одного CSV-файла ISSAI."""
    try:
        raw = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не удалось прочитать ISSAI CSV %s: %s", file_path, exc)
        return pd.Series(dtype="object")

    if "kk" in raw.columns:
        return raw["kk"]
    if "text" in raw.columns:
        return raw["text"]
    if "pair" in raw.columns:
        return raw["pair"].map(_extract_kk_from_pair)
    return pd.Series(dtype="object")


def load_issai_kazakh_ham(
    path: Path | str | None = None,
    max_rows: int = 500,
    min_len: int = 15,
    max_len: int = 220,
    random_state: int = 42,
) -> pd.DataFrame:
    """Загрузить безопасные казахские тексты из ``issai/kazsandra`` и ``issai/kazparc``.

    Так как эти корпуса не содержат разметки спама, отобранные строки
    маркируются как ``label=0`` и должны ограничиваться квотой при объединении.
    """
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = [
            HF_RAW_DIR / ISSAI_KAZSANDRA_FILENAME,
            HF_RAW_DIR / ISSAI_KAZPARC_FILENAME,
        ]

    series_parts: list[pd.Series] = []
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            extracted = _read_issai_csv(candidate)
            if not extracted.empty:
                series_parts.append(extracted)

    if not series_parts:
        return _empty_hf_frame()

    combined_texts = pd.concat(series_parts, ignore_index=True)
    cleaned = _filter_clean_texts(
        combined_texts,
        min_len=min_len,
        max_len=max_len,
        drop_web=True,
    ).drop_duplicates()
    if cleaned.empty:
        return _empty_hf_frame()

    df = pd.DataFrame(
        {
            "text": cleaned,
            "label": 0,
            "source": "issai",
            "language": "kk",
        }
    )
    sampled = _sample_rows(df, max_rows=max_rows, random_state=random_state)
    return _finalize_hf_frame(sampled)


def _resolve_kz_transformers_path(path: Path | str | None) -> Path | None:
    """Выбрать существующий локальный файл ``kz-transformers``."""
    if path is not None:
        candidate = Path(path)
        return candidate if candidate.is_file() and candidate.stat().st_size > 0 else None

    for filename in (KZ_TRANSFORMERS_SAMPLE_FILENAME, KZ_TRANSFORMERS_FILENAME):
        candidate = HF_RAW_DIR / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def load_kz_transformers_ham(
    path: Path | str | None = None,
    max_rows: int = 500,
    min_len: int = 15,
    max_len: int = 220,
    random_state: int = 42,
) -> pd.DataFrame:
    """Загрузить безопасные казахские тексты из ``kz-transformers/multidomain-kazakh-dataset``.

    Фильтрует записи по ``predicted_language == 'kaz'``, ``contains_kaz_symbols == 1``,
    длине ``min_len..max_len`` и отсутствию подстрок ``http`` и ``wikipedia``.
    """
    resolved = _resolve_kz_transformers_path(path)
    if resolved is None:
        return _empty_hf_frame()

    try:
        raw = pd.read_csv(resolved, encoding="utf-8", on_bad_lines="skip")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не удалось прочитать kz-transformers CSV %s: %s", resolved, exc)
        return _empty_hf_frame()

    required = {"text", "predicted_language", "contains_kaz_symbols"}
    if not required.issubset(raw.columns):
        return _empty_hf_frame()

    pred_lang = raw["predicted_language"].astype(str).str.strip().str.lower()
    kaz_sym_num = pd.to_numeric(raw["contains_kaz_symbols"], errors="coerce")
    kaz_sym_bool = raw["contains_kaz_symbols"].astype(str).str.strip().str.lower() == "true"
    mask = (pred_lang == "kaz") & ((kaz_sym_num == 1) | kaz_sym_bool)

    filtered_raw = raw.loc[mask]
    if filtered_raw.empty:
        return _empty_hf_frame()

    cleaned = _filter_clean_texts(
        filtered_raw["text"],
        min_len=min_len,
        max_len=max_len,
        drop_web=True,
    ).drop_duplicates()
    if cleaned.empty:
        return _empty_hf_frame()

    df = pd.DataFrame(
        {
            "text": cleaned,
            "label": 0,
            "source": "kz_transformers",
            "language": "kk",
        }
    )
    sampled = _sample_rows(df, max_rows=max_rows, random_state=random_state)
    return _finalize_hf_frame(sampled)


def _summarize_source_df(df: pd.DataFrame, status: str = "loaded", reason: str | None = None) -> dict[str, Any]:
    """Сформировать словарь статистики по отдельному источнику."""
    ham_count = int((df["label"] == 0).sum()) if not df.empty else 0
    spam_count = int((df["label"] == 1).sum()) if not df.empty else 0
    languages = (
        {str(lang): int(cnt) for lang, cnt in df["language"].value_counts().items()}
        if not df.empty
        else {}
    )
    entry: dict[str, Any] = {
        "status": status,
        "rows": int(len(df)),
        "ham": ham_count,
        "spam": spam_count,
        "languages": languages,
    }
    if reason is not None:
        entry["reason"] = reason
    return entry


def _maybe_download_missing_sources(download_missing: bool) -> dict[str, str]:
    """При необходимости скачать отсутствующие HF-файлы и вернуть причины сбоев."""
    if not download_missing:
        return {}

    failures: dict[str, str] = {}

    ruspam_dest = HF_RAW_DIR / RUSPAM_FILENAME
    if not ruspam_dest.is_file() or ruspam_dest.stat().st_size == 0:
        ok, msg = download_hf_file(
            "ruSpamModels/russian-spam-detection",
            "processed_combined.parquet",
            ruspam_dest,
        )
        if not ok:
            failures["ruSpamModels"] = msg

    tg_dest = HF_RAW_DIR / TELEGRAM_SPAM_FILENAME
    if not tg_dest.is_file() or tg_dest.stat().st_size == 0:
        ok, msg = download_hf_file(
            "AugustLight/telegram_spam_ru",
            "spam_dataset.jsonl",
            tg_dest,
        )
        if not ok:
            failures["telegram_spam_ru"] = msg

    issai_present = any(
        (HF_RAW_DIR / name).is_file() and (HF_RAW_DIR / name).stat().st_size > 0
        for name in (ISSAI_KAZSANDRA_FILENAME, ISSAI_KAZPARC_FILENAME)
    )
    if not issai_present:
        issai_errors: list[str] = []
        for repo_id, remote_path, filename in (
            ("issai/kazsandra", "polarity_classification/train_pc.csv", ISSAI_KAZSANDRA_FILENAME),
            ("issai/kazparc", "kazparc/08_kazparc_valid_kk_en.csv", ISSAI_KAZPARC_FILENAME),
        ):
            ok, msg = download_hf_file(
                repo_id,
                remote_path,
                HF_RAW_DIR / filename,
                max_bytes=_DEFAULT_SAMPLE_BYTES,
            )
            if not ok:
                issai_errors.append(f"{repo_id}: {msg}")
        if len(issai_errors) == 2:
            failures["issai"] = "; ".join(issai_errors)

    kz_present = any(
        (HF_RAW_DIR / name).is_file() and (HF_RAW_DIR / name).stat().st_size > 0
        for name in (KZ_TRANSFORMERS_SAMPLE_FILENAME, KZ_TRANSFORMERS_FILENAME)
    )
    if not kz_present:
        ok, msg = download_hf_file(
            "kz-transformers/multidomain-kazakh-dataset",
            "leipzig.csv",
            HF_RAW_DIR / KZ_TRANSFORMERS_SAMPLE_FILENAME,
            max_bytes=_DEFAULT_SAMPLE_BYTES,
        )
        if not ok:
            failures["kz_transformers"] = msg

    kazspam_dest = HF_RAW_DIR / KAZSPAM_V1_FILENAME
    if not kazspam_dest.is_file() or kazspam_dest.stat().st_size == 0:
        ok, msg = download_hf_file(
            "salyamq/kazspam-v1",
            "classified_spam_10k_fixed.jsonl",
            kazspam_dest,
        )
        if not ok:
            failures["kazspam_v1"] = msg

    return failures


def _allocate_kz_ham_quotas(
    issai_df: pd.DataFrame,
    kz_df: pd.DataFrame,
    cap: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Распределить квоту ``cap`` между доступными источниками казахского ham."""
    if cap <= 0:
        return _empty_hf_frame(), _empty_hf_frame()

    has_issai = not issai_df.empty
    has_kz = not kz_df.empty
    if has_issai and has_kz:
        half_cap = cap // 2
        other_half = cap - half_cap
        # Если один источник меньше своей половины, остаток квоты передаётся второму.
        if len(issai_df) < half_cap:
            issai_take = len(issai_df)
            kz_take = cap - issai_take
        elif len(kz_df) < other_half:
            kz_take = len(kz_df)
            issai_take = cap - kz_take
        else:
            issai_take = half_cap
            kz_take = other_half
        return (
            _finalize_hf_frame(_sample_rows(issai_df, issai_take, random_state=random_state)),
            _finalize_hf_frame(_sample_rows(kz_df, kz_take, random_state=random_state)),
        )
    if has_issai:
        return (
            _finalize_hf_frame(_sample_rows(issai_df, cap, random_state=random_state)),
            _empty_hf_frame(),
        )
    if has_kz:
        return (
            _empty_hf_frame(),
            _finalize_hf_frame(_sample_rows(kz_df, cap, random_state=random_state)),
        )
    return _empty_hf_frame(), _empty_hf_frame()


def load_combined_sms_dataset(
    download_missing: bool = False,
    ruspam_max_ham: int = 15000,
    ruspam_max_spam: int = 10000,
    kz_ham_ratio: float = 1.5,
    min_kz_ham: int = 200,
    max_kz_ham: int = 1000,
    output_csv_path: Path | str | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Объединить UCI ``SMSSpamCollection`` и доступные наборы HuggingFace.

    Parameters
    ----------
    download_missing:
        Если ``True``, пытается скачать отсутствующие файлы в ``HF_RAW_DIR``.
        Ошибки 401/403 для закрытых наборов не приводят к падению, а отражаются
        в отчёте.
    ruspam_max_ham:
        Максимум ham-строк из ``ruSpamModels``.
    ruspam_max_spam:
        Максимум spam-строк из ``ruSpamModels``.
    kz_ham_ratio:
        Коэффициент квотирования казахских ham-строк из неразмеченных
        источников (``issai``, ``kz_transformers``) относительно числа
        найденных казахских spam-строк.
    min_kz_ham:
        Нижняя граница квоты казахских ham-строк из неразмеченных источников.
    max_kz_ham:
        Верхняя граница квоты казахских ham-строк из неразмеченных источников.
    output_csv_path:
        Необязательный путь для сохранения итогового CSV (UTF-8, ``index=False``).
    random_state:
        Зерно генератора случайных чисел для воспроизводимого сэмплирования.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, Any]]
        Объединённый ``DataFrame`` со столбцами ``["id", "text", "label", "source", "language"]``
        и словарь-отчёт ``dataset_sources_report``.
    """
    download_failures = _maybe_download_missing_sources(download_missing)

    uci_raw = load_sms_dataset()
    uci_df = _finalize_hf_frame(
        pd.DataFrame(
            {
                "text": uci_raw["text"],
                "label": uci_raw["label"],
                "source": "uci_sms",
                "language": "en",
            }
        )
    )

    ruspam_df = load_ruspam_dataset(
        max_ham=ruspam_max_ham,
        max_spam=ruspam_max_spam,
        random_state=random_state,
    )
    telegram_df = load_telegram_spam_ru(random_state=random_state)
    kazspam_df = load_kazspam_v1(random_state=random_state)

    labeled_parts = [uci_df, ruspam_df, telegram_df, kazspam_df]
    labeled_combined = pd.concat(labeled_parts, ignore_index=True)
    kz_spam_count = int(
        ((labeled_combined["language"] == "kk") & (labeled_combined["label"] == 1)).sum()
    )

    cap = min(max_kz_ham, max(min_kz_ham, int(kz_spam_count * kz_ham_ratio)))
    issai_pool = load_issai_kazakh_ham(max_rows=max(cap, 1), random_state=random_state)
    kz_pool = load_kz_transformers_ham(max_rows=max(cap, 1), random_state=random_state)
    issai_df, kz_df = _allocate_kz_ham_quotas(
        issai_pool,
        kz_pool,
        cap=cap,
        random_state=random_state,
    )

    source_frames: dict[str, pd.DataFrame] = {
        "uci_sms": uci_df,
        "ruSpamModels": ruspam_df,
        "telegram_spam_ru": telegram_df,
        "kazspam_v1": kazspam_df,
        "issai": issai_df,
        "kz_transformers": kz_df,
    }

    sources_report: dict[str, dict[str, Any]] = {}
    skipped_sources: dict[str, str] = {}
    active_parts: list[pd.DataFrame] = []

    for source_name, part_df in source_frames.items():
        if part_df.empty:
            reason = download_failures.get(source_name, "file missing or empty after filtering")
            sources_report[source_name] = _summarize_source_df(
                part_df,
                status="skipped",
                reason=reason,
            )
            skipped_sources[source_name] = reason
        else:
            sources_report[source_name] = _summarize_source_df(part_df, status="loaded")
            active_parts.append(part_df)

    merged = (
        pd.concat(active_parts, ignore_index=True)
        if active_parts
        else _empty_hf_frame()
    )
    merged.insert(0, "id", pd.Series(range(1, len(merged) + 1), dtype="int64"))
    merged = merged.loc[:, list(COMBINED_COLUMNS)].astype(
        {
            "id": "int64",
            "text": "object",
            "label": "int64",
            "source": "object",
            "language": "object",
        }
    )

    if output_csv_path is not None:
        dest = Path(output_csv_path).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(dest, index=False, encoding="utf-8")

    report: dict[str, Any] = {
        "sources": sources_report,
        "skipped_sources": skipped_sources,
        "total_rows": int(len(merged)),
        "total_ham": int((merged["label"] == 0).sum()) if not merged.empty else 0,
        "total_spam": int((merged["label"] == 1).sum()) if not merged.empty else 0,
        "kz_spam_count": kz_spam_count,
        "kz_ham_cap": int(cap),
        "kz_ham_added": int(len(issai_df) + len(kz_df)),
    }
    return merged, report
