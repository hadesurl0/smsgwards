"""Модульные тесты для загрузчика HuggingFace-датасетов (ml/data/load_hf_datasets.py).

Все тесты работают исключительно на временных файлах ``tmp_path`` и моках
``monkeypatch`` без реальных сетевых запросов.
"""

from __future__ import annotations

import io
import json
import logging
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

import ml.data.load_hf_datasets as hf_mod
from ml.data.load_hf_datasets import (
    COMBINED_COLUMNS,
    STANDARD_COLUMNS,
    download_hf_file,
    get_hf_token,
    load_combined_sms_dataset,
    load_issai_kazakh_ham,
    load_kazspam_v1,
    load_kz_transformers_ham,
    load_ruspam_dataset,
    load_telegram_spam_ru,
)


def _fake_uci_df() -> pd.DataFrame:
    """Создать небольшой синтетический аналог UCI SMSSpamCollection."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": [
                "Hey, are we still meeting for lunch today?",
                "WINNER!! Claim your free prize now at http://example.com",
                "Ok see you tomorrow morning at nine.",
            ],
            "label": [0, 1, 0],
        }
    ).astype({"id": "int64", "text": "object", "label": "int64"})


def test_get_hf_token_priority_and_no_secret_in_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Проверка приоритета источников HF-токена и отсутствия токена в логах."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert get_hf_token() is None

    token_file = tmp_path / ".cache" / "huggingface" / "token"
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text("hf_cached_secret_token\n", encoding="utf-8")
    assert get_hf_token() == "hf_cached_secret_token"

    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_hub_env_token")
    assert get_hf_token() == "hf_hub_env_token"

    monkeypatch.setenv("HF_TOKEN", "hf_primary_env_token")
    with caplog.at_level(logging.DEBUG):
        assert get_hf_token() == "hf_primary_env_token"

    assert "hf_primary_env_token" not in caplog.text
    assert "hf_cached_secret_token" not in caplog.text


def test_load_ruspam_dataset_balancing_and_kazakh(tmp_path: Path):
    """Парсинг Parquet ruSpamModels: фильтр длины, балансировка и язык kk."""
    parquet_path = tmp_path / "russian_spam_detection.parquet"
    df_raw = pd.DataFrame(
        {
            "message": [
                "Привет, как твои дела сегодня вечером?",
                "Обычное сообщение о встрече в офисе завтра",
                "Ещё одно нормальное русское сообщение без спама",
                "Сәлем достым, ертең кездесейік пе?",
                "short",  # слишком короткое (< 10)
                "Обычное сообщение о встрече в офисе завтра",  # дубликат
                "Срочно перейдите по ссылке и получите миллион рублей",
                "Вы выиграли крупный денежный приз сегодня",
                "Құттықтаймыз! Сіз ұтыс ойынында жеңімпаз атандыңыз",
                "Шұғыл түрде картаңыздың кодын жіберіңіз",
            ],
            "label": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    df_raw.to_parquet(parquet_path, index=False)

    balanced = load_ruspam_dataset(
        path=parquet_path,
        max_ham=2,
        max_spam=3,
        include_kazakh_rows=True,
        min_len=10,
        max_len=400,
        random_state=42,
    )
    assert list(balanced.columns) == list(STANDARD_COLUMNS)
    assert (balanced["label"] == 0).sum() == 2
    assert (balanced["label"] == 1).sum() == 3
    assert set(balanced["source"].unique()) == {"ruSpamModels"}
    assert "kk" in set(balanced["language"])
    assert "ru" in set(balanced["language"])
    assert "short" not in set(balanced["text"])
    assert balanced["text"].is_unique

    ru_only = load_ruspam_dataset(
        path=parquet_path,
        max_ham=10,
        max_spam=10,
        include_kazakh_rows=False,
        min_len=10,
        max_len=400,
        random_state=42,
    )
    assert set(ru_only["language"].unique()) == {"ru"}


def test_load_telegram_spam_ru_jsonl(tmp_path: Path):
    """Парсинг JSONL telegram_spam_ru с пропуском битых строк и дубликатов."""
    jsonl_path = tmp_path / "telegram_spam_ru.jsonl"
    lines = [
        json.dumps({"text": "Подписывайся на закрытый канал с сигналами", "label": 1}, ensure_ascii=False),
        "",
        "{invalid json",
        json.dumps({"text": "Завтра лекция начнётся в 10 утра", "label": 0}, ensure_ascii=False),
        json.dumps({"text": "Завтра лекция начнётся в 10 утра", "label": 0}, ensure_ascii=False),
        json.dumps({"text": "   ", "label": 1}, ensure_ascii=False),
    ]
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    df = load_telegram_spam_ru(path=jsonl_path)
    assert list(df.columns) == list(STANDARD_COLUMNS)
    assert len(df) == 2
    assert df["label"].tolist() == [1, 0]
    assert set(df["source"]) == {"telegram_spam_ru"}
    assert set(df["language"]) == {"ru"}


def test_load_issai_kazakh_ham_text_kk_and_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Парсинг issai CSV: колонки text, kk и pair как label=0, язык kk."""
    monkeypatch.setattr(hf_mod, "HF_RAW_DIR", tmp_path)

    kazsandra_path = tmp_path / "issai_kazsandra.csv"
    pd.DataFrame(
        {
            "text": [
                "Бүгін ауа райы өте тамаша болды!",
                "Қысқа",  # < 15 символов
                "Мына сілтемені қараңыз http://spam.example.kz",  # содержит http
            ]
        }
    ).to_csv(kazsandra_path, index=False, encoding="utf-8")

    kazparc_path = tmp_path / "issai_kazparc.csv"
    pd.DataFrame(
        {
            "kk": [
                "Қазақстан тарихы мен мәдениеті өте бай.",
                "Алматы қаласында жаңа саябақ ашылды.",
            ]
        }
    ).to_csv(kazparc_path, index=False, encoding="utf-8")

    df_combined = load_issai_kazakh_ham(max_rows=10, min_len=15, max_len=220, random_state=42)
    assert list(df_combined.columns) == list(STANDARD_COLUMNS)
    assert len(df_combined) == 3
    assert set(df_combined["label"]) == {0}
    assert set(df_combined["source"]) == {"issai"}
    assert set(df_combined["language"]) == {"kk"}

    pair_path = tmp_path / "issai_pair.csv"
    pd.DataFrame(
        {
            "pair": [
                json.dumps({"kk": "Бұл жүйе өте ыңғайлы жұмыс істейді.", "en": "This system works well."}, ensure_ascii=False),
            ]
        }
    ).to_csv(pair_path, index=False, encoding="utf-8")

    df_pair = load_issai_kazakh_ham(path=pair_path, max_rows=5, min_len=15, max_len=220)
    assert len(df_pair) == 1
    assert df_pair["text"].iloc[0] == "Бұл жүйе өте ыңғайлы жұмыс істейді."
    assert df_pair["label"].iloc[0] == 0


def test_load_kz_transformers_ham_filters(tmp_path: Path):
    """Фильтрация kz-transformers CSV по языку, казахским символам, длине и URL."""
    csv_path = tmp_path / "kz_transformers_leipzig_sample.csv"
    long_text = "Қазақ тіліндегі өте ұзын сөйлем " * 12
    pd.DataFrame(
        {
            "text": [
                "Қазақстанда білім беру саласы қарқынды дамуда.",
                "Астанада жаңа технологиялық орталық ашылды.",
                "Это русский текст с predicted_language rus",
                "Kazakh without special letters here",
                "Қысқа сөз",  # короче 15
                long_text,  # длиннее 220
                "Толығырақ осында: https://example.kz/news",  # http
                "Дереккөз: Wikipedia мақаласынан алынған",  # wikipedia
            ],
            "predicted_language": ["kaz", "kaz", "rus", "kaz", "kaz", "kaz", "kaz", "kaz"],
            "contains_kaz_symbols": [1, 1, 0, 0, 1, 1, 1, 1],
        }
    ).to_csv(csv_path, index=False, encoding="utf-8")

    df = load_kz_transformers_ham(
        path=csv_path,
        max_rows=10,
        min_len=15,
        max_len=220,
        random_state=42,
    )
    assert list(df.columns) == list(STANDARD_COLUMNS)
    assert len(df) == 2
    assert set(df["label"]) == {0}
    assert set(df["source"]) == {"kz_transformers"}
    assert set(df["language"]) == {"kk"}


def test_download_hf_file_cached_csv_trim_and_403(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Проверка кэша, Range-обрезания CSV и безопасной обработки 401/403."""
    cached = tmp_path / "cached.csv"
    cached.write_text("text\nhello\n", encoding="utf-8")
    ok_cached, status_cached = download_hf_file("org/repo", "file.csv", cached)
    assert (ok_cached, status_cached) == (True, "cached")

    monkeypatch.setenv("HF_TOKEN", "hf_super_secret_do_not_log")
    captured_headers: dict[str, str] = {}

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def fake_urlopen_ok(request, timeout=120):  # noqa: ARG001
        captured_headers.update({k.lower(): v for k, v in request.header_items()})
        return FakeResponse(b"col1,col2\nrow1,ok\nincomplete_row")

    monkeypatch.setattr(hf_mod.urllib.request, "urlopen", fake_urlopen_ok)
    csv_dest = tmp_path / "sample.csv"
    ok_dl, status_dl = download_hf_file(
        "kz-transformers/multidomain-kazakh-dataset",
        "leipzig.csv",
        csv_dest,
        max_bytes=64,
    )
    assert (ok_dl, status_dl) == (True, "downloaded")
    assert captured_headers.get("range") == "bytes=0-63"
    assert captured_headers.get("authorization") == "Bearer hf_super_secret_do_not_log"
    assert csv_dest.read_text(encoding="utf-8") == "col1,col2\nrow1,ok\n"

    def fake_urlopen_403(request, timeout=120):  # noqa: ARG001
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(hf_mod.urllib.request, "urlopen", fake_urlopen_403)
    gated_dest = tmp_path / "gated.csv"
    with caplog.at_level(logging.DEBUG):
        ok_403, msg_403 = download_hf_file("issai/kazsandra", "train_pc.csv", gated_dest)

    assert ok_403 is False
    assert msg_403.startswith("401/403:")
    assert not gated_dest.exists()
    assert list(tmp_path.glob("*.part")) == []
    assert "hf_super_secret_do_not_log" not in caplog.text


def test_load_combined_sms_dataset_handles_missing_and_403(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """При 403/отсутствующих файлах объединение не падает и заполняет отчёт."""
    monkeypatch.setattr(hf_mod, "HF_RAW_DIR", tmp_path)
    monkeypatch.setattr(hf_mod, "load_sms_dataset", _fake_uci_df)

    def fake_urlopen_403(request, timeout=120):  # noqa: ARG001
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=403,
            msg="Gated dataset",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(hf_mod.urllib.request, "urlopen", fake_urlopen_403)

    df, report = load_combined_sms_dataset(download_missing=True, random_state=42)
    assert list(df.columns) == list(COMBINED_COLUMNS)
    assert len(df) == 3
    assert df["id"].tolist() == [1, 2, 3]
    assert set(df["source"]) == {"uci_sms"}

    assert report["sources"]["uci_sms"]["status"] == "loaded"
    for missing_src in ("ruSpamModels", "telegram_spam_ru", "issai", "kz_transformers", "kazspam_v1"):
        assert report["sources"][missing_src]["status"] == "skipped"
        assert "401/403" in report["skipped_sources"][missing_src]


def test_load_combined_sms_dataset_kz_ham_ratio_and_output_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Квотирование казахских ham-строк пропорционально казахскому спаму и запись CSV."""
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(hf_mod, "HF_RAW_DIR", hf_dir)
    monkeypatch.setattr(hf_mod, "load_sms_dataset", _fake_uci_df)

    # 4 казахских spam-сообщения -> при kz_ham_ratio=1.5 квота cap = 6 (по 3 из issai и kz_transformers).
    pd.DataFrame(
        {
            "message": [
                "Обычное длинное русское сообщение номер один",
                "Құттықтаймыз! Сіз сыйлық ұтып алдыңыз, сілтемеге өтіңіз",
                "Шұғыл: банк картаңыз бұғатталды, кодты хабарлаңыз",
                "Сізге арнайы несие мақұлданды, өтінімді растаңыз",
                "Тегін ақша алу үшін жеке деректеріңізді жіберіңіз",
            ],
            "label": [0, 1, 1, 1, 1],
        }
    ).to_parquet(hf_dir / "russian_spam_detection.parquet", index=False)

    pd.DataFrame(
        {
            "text": [
                f"Қазақша бейтарап мәтін issai нөмірі {i} тексеру үшін"
                for i in range(10)
            ]
        }
    ).to_csv(hf_dir / "issai_kazsandra.csv", index=False, encoding="utf-8")

    pd.DataFrame(
        {
            "text": [
                f"Қазақша жаңалық мәтіні leipzig нөмірі {i} тексеру үшін"
                for i in range(10)
            ],
            "predicted_language": ["kaz"] * 10,
            "contains_kaz_symbols": [1] * 10,
        }
    ).to_csv(hf_dir / "kz_transformers_leipzig_sample.csv", index=False, encoding="utf-8")

    out_csv = tmp_path / "out" / "combined_sms.csv"
    df, report = load_combined_sms_dataset(
        download_missing=False,
        kz_ham_ratio=1.5,
        min_kz_ham=2,
        max_kz_ham=10,
        output_csv_path=out_csv,
        random_state=42,
    )

    assert list(df.columns) == list(COMBINED_COLUMNS)
    assert df["id"].dtype == "int64"
    assert df["label"].dtype == "int64"
    assert df["id"].tolist() == list(range(1, len(df) + 1))

    assert report["kz_spam_count"] == 4
    assert report["kz_ham_cap"] == 6
    assert report["kz_ham_added"] == 6
    assert report["sources"]["issai"]["rows"] == 3
    assert report["sources"]["kz_transformers"]["rows"] == 3
    assert report["sources"]["telegram_spam_ru"]["status"] == "skipped"

    assert out_csv.is_file()
    saved = pd.read_csv(out_csv, encoding="utf-8")
    assert list(saved.columns) == list(COMBINED_COLUMNS)
    assert len(saved) == len(df)


def test_load_kazspam_v1_filters_length_and_category(tmp_path: Path):
    """Загрузчик salyamq/kazspam-v1 отбирает короткие казахские casino/porn как spam."""
    jsonl_path = tmp_path / "kazspam_v1.jsonl"
    records = [
        {"classification": "casino", "text": "Қазақстандық казино бонус 500 тегін фриспин ал!"},
        {"classification": "casino", "text": "Эксклюзивті слоттар қолма-қол ақша ұтып ал " + ("қазақша " * 120)},
        {"classification": "casino", "text": "Only English casino text without Kazakh symbols here"},
        {"classification": "porn", "text": "Тегін видео онлайн тіркелусіз қарау үшін сілтемеге өтіңіз"},
        {"classification": "normal", "text": "Алматыда жаңа мектеп ашылды және көрме өтті"},
        {"classification": "other", "text": "Басқа жаңалықтар топтамасы бүгін жарияланды"},
    ]
    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    df = load_kazspam_v1(
        jsonl_path,
        max_casino=10,
        max_porn=10,
        max_casino_len=120,
        max_porn_len=120,
        random_state=42,
    )

    assert list(df.columns) == list(STANDARD_COLUMNS)
    assert len(df) == 2
    assert (df["label"] == 1).all()
    assert (df["source"] == "kazspam_v1").all()
    assert (df["language"] == "kk").all()
    assert any("фриспин" in t for t in df["text"])
    assert any("бейне" in t or "видео" in t for t in df["text"])
