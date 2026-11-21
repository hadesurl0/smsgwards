"""Интеграционные тесты подготовки объединённого датасета и smoke-проверка predict_sms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml import train_sms
from ml.features import build_sms_feature_union
from ml.preprocessing import normalize_sms


def _make_sample_combined_df() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ham_samples = [
        ("Hey, are we still meeting at 1pm tomorrow?", "uci_sms", "en"),
        ("Ok, call me when you get home safely.", "uci_sms", "en"),
        ("Привет, во сколько завтра встречаемся у офиса?", "ruSpamModels", "ru"),
        ("Мама, я уже доехал, всё хорошо, скоро буду.", "ruSpamModels", "ru"),
        ("Сәлем, ертең сағат онда кездесейік пе?", "kz_transformers", "kk"),
        ("Рақмет, құжаттарды электронды поштаға жібердім.", "kz_transformers", "kk"),
        ("Please bring the documents to room 204.", "uci_sms", "en"),
        ("Добрый день, заказ доставлен в пункт выдачи.", "ruSpamModels", "ru"),
        ("Жиын сағат үште басталады, кешікпеңіз.", "kz_transformers", "kk"),
        ("Thanks for the update, talk soon.", "uci_sms", "en"),
    ]
    spam_samples = [
        ("URGENT! Your bank card is blocked, verify at http://bit.ly/fake", "uci_sms", "en"),
        ("WIN £1000 cash prize now! Reply CLAIM to 88088", "uci_sms", "en"),
        ("СРОЧНО! Карта заблокирована, перейдите по ссылке http://evil.ru", "ruSpamModels", "ru"),
        ("Легкий заработок от 50000 руб в день, пиши в ЛС прямо сейчас!", "ruSpamModels", "ru"),
        ("Тегін 500000 теңге ұтып ал, сілтемеге өт http://scam.kz", "ruSpamModels", "kk"),
        ("FREE entry to win an iPhone! Click http://promo.example", "uci_sms", "en"),
        ("Ваш аккаунт взломан, срочно подтвердите код из СМС!", "ruSpamModels", "ru"),
        ("Шұғыл! Картаңыз бұғатталды, кодты хабарлаңыз http://phish.kz", "ruSpamModels", "kk"),
        ("Exclusive casino bonus 500$ waiting for you now!", "uci_sms", "en"),
        ("Пассивный доход без вложений, подробности в телеграм @spam", "ruSpamModels", "ru"),
    ]
    idx = 1
    for repeat in range(2):
        for text, source, lang in ham_samples:
            rows.append(
                {
                    "id": idx,
                    "text": f"{text} ({repeat})" if repeat else text,
                    "label": 0,
                    "source": source,
                    "language": lang,
                }
            )
            idx += 1
        for text, source, lang in spam_samples:
            rows.append(
                {
                    "id": idx,
                    "text": f"{text} ({repeat})" if repeat else text,
                    "label": 1,
                    "source": source,
                    "language": lang,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def _patch_train_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    raw_path = processed / "sms_raw.csv"
    monkeypatch.setattr(train_sms, "PROCESSED_DIR", processed)
    monkeypatch.setattr(train_sms, "RAW_CSV_PATH", raw_path)
    monkeypatch.setattr(train_sms, "CLEAN_CSV_PATH", processed / "sms_clean.csv")
    monkeypatch.setattr(train_sms, "CLEAN_REPORT_PATH", processed / "sms_cleaning_report.json")
    monkeypatch.setattr(train_sms, "SPLIT_JSON_PATH", processed / "sms_split.json")
    return raw_path


def test_prepare_clean_data_rebuilds_legacy_raw_without_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path = _patch_train_paths(monkeypatch, tmp_path)
    pd.DataFrame(
        {
            "id": [1, 2],
            "text": ["hello friend", "win prize http://x.ru"],
            "label": [0, 1],
        }
    ).to_csv(raw_path, index=False, encoding="utf-8")

    sample_df = _make_sample_combined_df()
    calls: list[bool] = []

    def fake_loader(**kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        calls.append(bool(kwargs.get("download_missing")))
        out_path = Path(kwargs["output_csv_path"])
        sample_df.to_csv(out_path, index=False, encoding="utf-8")
        return sample_df.copy(), {"counts_by_source": {"uci_sms": 16, "ruSpamModels": 18, "kz_transformers": 6}}

    monkeypatch.setattr(train_sms, "load_combined_sms_dataset", fake_loader)

    cleaned_df, report, splits = train_sms.prepare_clean_data_and_splits(
        force_reload=False,
        download_missing=False,
    )

    assert calls == [False]
    assert "source" in cleaned_df.columns
    assert report["dataset_sources"]["counts_by_source"]["ruSpamModels"] == 18
    assert train_sms.CLEAN_CSV_PATH.exists()
    assert train_sms.SPLIT_JSON_PATH.exists()
    total_split = (
        len(splits["train_indices"])
        + len(splits["val_indices"])
        + len(splits["test_indices"])
    )
    assert total_split == len(cleaned_df)


def test_prepare_clean_data_reuses_existing_combined_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path = _patch_train_paths(monkeypatch, tmp_path)
    sample_df = _make_sample_combined_df()
    sample_df.to_csv(raw_path, index=False, encoding="utf-8")

    def fail_if_called(**kwargs: Any) -> Any:
        raise AssertionError("load_combined_sms_dataset should not be called when source column exists")

    monkeypatch.setattr(train_sms, "load_combined_sms_dataset", fail_if_called)

    cleaned_df, report, splits = train_sms.prepare_clean_data_and_splits(force_reload=False)

    assert len(cleaned_df) == len(sample_df)
    assert "clean_counts_by_source" in report["dataset_sources"]
    assert set(report["dataset_sources"]["clean_counts_by_source"]) == {
        "uci_sms",
        "ruSpamModels",
        "kz_transformers",
    }
    saved_splits = json.loads(train_sms.SPLIT_JSON_PATH.read_text(encoding="utf-8"))
    assert saved_splits["train_indices"] == splits["train_indices"]


def test_predict_sms_smoke_with_multilingual_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _make_sample_combined_df()
    texts = [normalize_sms(t) for t in df["text"]]
    labels = df["label"].to_numpy()

    pipe = Pipeline(
        [
            ("features", build_sms_feature_union()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        ]
    )
    pipe.fit(texts, labels)

    model_path = tmp_path / "sms_model.joblib"
    joblib.dump(pipe, model_path)

    monkeypatch.setattr(train_sms, "_CACHED_MODEL", None)
    monkeypatch.setattr(train_sms, "_CACHED_MODEL_PATH", None)
    monkeypatch.setattr(train_sms, "FINAL_META_PATH", tmp_path / "missing_meta.json")

    spam_res = train_sms.predict_sms(
        "Сpочно! Кaрта заблокирована, перейдите на http://scam.ru",
        model_path=model_path,
        threshold=0.5,
    )
    ham_res = train_sms.predict_sms(
        "Привет, во сколько завтра встречаемся у офиса?",
        model_path=model_path,
        threshold=0.5,
    )
    kk_res = train_sms.predict_sms(
        "Сәлем, ертең сағат онда кездесейік пе?",
        model_path=model_path,
        threshold=0.5,
    )

    assert spam_res["label"] == 1
    assert spam_res["category"] == "FRAUDULENT"
    assert "срочно" in spam_res["text_clean"]
    assert "карта" in spam_res["text_clean"]
    assert 0.0 <= spam_res["probability"] <= 1.0
    assert ham_res["language"] == "ru"
    assert kk_res["language"] == "kk"


@pytest.mark.django_db
def test_trained_languages_suppresses_legacy_warning_and_keeps_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Для обученных языков (ru/kk/en) v2.0.0 не выдаёт ложное предупреждение ФТ-2.8."""
    from apps.analysis import services as analysis_services
    from apps.analysis.models import Check
    from ml.predictor import ModelPredictor
    from ml.rules import detect_se_rules

    predictor = ModelPredictor.get_instance()
    assert {"en", "ru", "kk"}.issubset(predictor.trained_languages)

    ru_res = analysis_services.analyze_sms("Привет, встречаемся завтра в шесть вечера.")
    assert ru_res.language == "ru"
    assert ru_res.language_warning == ""
    assert not any("ФТ-2.8" in reason for reason in ru_res.explanation)

    kk_res = analysis_services.analyze_sms("Сәлем, ертең сағат онда кездесейік пе?")
    assert kk_res.language == "kk"
    assert kk_res.language_warning == ""
    assert not any("ФТ-2.8" in reason for reason in kk_res.explanation)

    # Проверка Check.language_warning для v2.0.0 vs v1.0.0
    check_v2 = Check(check_type=Check.TYPE_SMS, language="kk", sms_model_version="2.0.0", input_masked="x", recommendation="r")
    check_v1 = Check(check_type=Check.TYPE_SMS, language="kk", sms_model_version="1.0.0", input_masked="x", recommendation="r")
    assert check_v2.language_warning == ""
    assert check_v1.language_warning != ""

    # Казахские правила СИ (1414 / egov.kz / растау коды / бұғатталады)
    se = detect_se_rules("1414 egov.kz: картаңыз бұғатталады, растау коды хабарлаңыз")
    codes = {g["code"] for g in se}
    assert {"AUTHORITY", "ACCOUNT_BLOCK", "CODE_REQUEST"}.issubset(codes)

    # Симулируем язык 'other' — предупреждение ФТ-2.8 должно сработать
    monkeypatch.setattr("ml.preprocessing.detect_language", lambda _t: "other")
    other_res = analysis_services.analyze_sms("123456 7890")
    assert other_res.language == "other"
    assert other_res.language_warning != ""
    assert any("ФТ-2.8" in reason for reason in other_res.explanation)


@pytest.mark.django_db
def test_dashboard_active_models_info_filters_nested_metrics() -> None:
    """_load_active_models_info возвращает актуальную v2.0.0 без вложенных dict вроде confusion_matrix."""
    from django.utils import timezone
    from apps.analysis.models import ModelVersion
    from apps.dashboard.views import _load_active_models_info

    # Создаём устаревшую запись 1.0.0 в БД — функция должна подтянуть свежую 2.0.0 с диска
    ModelVersion.objects.create(
        model_type="sms",
        version="1.0.0",
        algorithm="Old",
        metrics_json={"f1": 0.5},
        threshold=0.5,
        trained_at=timezone.now(),
        is_active=True,
    )
    models_info = _load_active_models_info()
    sms_entry = next(m for m in models_info if m["model_type"] == "sms")
    assert sms_entry["version"] == "2.0.0"
    assert "confusion_matrix" not in sms_entry["metrics"]
    assert all(isinstance(v, (int, float)) for v in sms_entry["metrics"].values())

