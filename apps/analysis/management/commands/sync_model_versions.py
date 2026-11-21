"""Синхронизация таблицы ModelVersion в БД с метаданными обученных ML-артефактов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from apps.analysis.models import ModelVersion


def _extract_scalar_metrics(raw: Any) -> dict[str, float]:
    """Оставить только скалярные числовые метрики."""
    if not isinstance(raw, dict):
        return {}
    return {
        str(k): round(float(v), 4)
        for k, v in raw.items()
        if k != "threshold" and isinstance(v, (int, float)) and not isinstance(v, bool)
    }


class Command(BaseCommand):
    help = "Синхронизировать активные версии ModelVersion с ml/artifacts/*_meta.json"

    def handle(self, *args: Any, **options: Any) -> None:
        artifacts_dir = Path(settings.BASE_DIR) / "ml" / "artifacts"
        specs = (
            ("sms", "sms_model_meta.json", "Logistic Regression (balanced)"),
            ("url", "url_model_meta.json", "HistGradientBoosting"),
        )

        for model_type, meta_name, fallback_algo in specs:
            meta_path = artifacts_dir / meta_name
            if not meta_path.is_file():
                self.stdout.write(
                    self.style.WARNING(f"Пропущено: файл {meta_path} не найден.")
                )
                continue

            data = json.loads(meta_path.read_text(encoding="utf-8"))
            version = str(data.get("version", "1.0.0"))
            algorithm = str(data.get("algorithm", data.get("model_name", fallback_algo)))
            threshold = float(data.get("threshold", data.get("t2", 0.5)))
            metrics = _extract_scalar_metrics(data.get("test_metrics") or data.get("metrics"))
            raw_dt = data.get("trained_at") or data.get("date")
            trained_at = parse_datetime(str(raw_dt)) if raw_dt else None
            if trained_at is None:
                trained_at = timezone.now()
            elif timezone.is_naive(trained_at):
                trained_at = timezone.make_aware(trained_at)

            ModelVersion.objects.filter(model_type=model_type, is_active=True).update(
                is_active=False
            )
            ModelVersion.objects.create(
                model_type=model_type,
                version=version,
                algorithm=algorithm,
                metrics_json=metrics,
                threshold=threshold,
                trained_at=trained_at,
                is_active=True,
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"Обновлена модель {model_type.upper()}: v{version} ({algorithm}, threshold={threshold})"
                )
            )
