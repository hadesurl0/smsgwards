# SMSGwards — Интеллектуальная система обнаружения мошеннических SMS-сообщений и подозрительных ссылок

**SMSGwards** — веб-приложение на базе Django 5, Django REST Framework и методов машинного обучения (`scikit-learn`), предназначенное для выявления мошеннических SMS (смишинга), признаков социальной инженерии и фишинговых интернет-ссылок (URL).

- **Авторы проекта:** Мурат Ринат, Мирланов Айдархан
- **Репозиторий:** `https://github.com/hadesurl0/smsgwards`
- **Презентация проекта:** [`docs/SMSGwards_Presentation.pptx`](docs/SMSGwards_Presentation.pptx) / [`docs/SMSGwards_Presentation.pdf`](docs/SMSGwards_Presentation.pdf)

> **Дисклеймер:** Оценка системы носит вероятностный (рекомендательный) характер и не гарантирует безопасность сообщения или ссылки. Проверка ссылок выполняется исключительно по структуре строки без сетевого открытия страниц.

---

## Быстрый запуск проекта (6 команд)

```bash
# 1. Перейти в директорию проекта и активировать виртуальное окружение
cd smsgwards && python3 -m venv .venv && source .venv/bin/activate

# 2. Установить зависимости с зафиксированными версиями
pip install -r requirements.txt

# 3. Создать файл конфигурации окружения из шаблона
cp .env.example .env

# 4. Применить миграции базы данных
python manage.py migrate

# 5. Заполнить БД демонстрационными пользователями и 35 проверками за 30 дней
python manage.py seed_demo

# 6. Запустить веб-сервер разработки
python manage.py runserver
```

После запуска откройте в браузере: `http://127.0.0.1:8000/`

### Демонстрационные учётные записи
| Роль | Логин | Пароль | Назначение |
|---|---|---|---|
| **Пользователь** | `demo_user` | `DemoPass123!` | Проверка SMS и URL, личный Dashboard, история (35 записей) |
| **Администратор** | `admin` | `AdminPass123!` | Общесистемная агрегированная статистика без чужих текстов SMS, метрики активных моделей |

---

## Структура репозитория

```text
smsgwards/
├── config/                 # Настройки Django (settings.py, urls.py, exceptions.py, wsgi.py)
├── apps/
│   ├── accounts/           # Регистрация, вход (блокировка после 5 попыток), профиль
│   ├── analysis/           # Сервис совместного анализа SMS+URL, модели БД, REST API, история
│   └── dashboard/          # Панель мониторинга, статистика (агрегация в БД)
├── ml/
│   ├── preprocessing.py    # Очистка и нормализация SMS, определение языка, маскирование
│   ├── url_features.py     # Извлечение 22 структурных признаков URL без сетевых запросов
│   ├── rules.py            # Словарь признаков социальной инженерии (ru, kk, en)
│   ├── risk.py             # Расчёт Risk Score (0–100), правила усиления, объяснения
│   ├── predictor.py        # Синглтон загрузки обученных моделей
│   └── artifacts/          # Сохранённые модели (sms_model.joblib, url_model.joblib) и метаданные
├── templates/              # HTML5-шаблоны (Bootstrap 5 + Chart.js, изумрудно-бирюзовая тема)
├── static/                 # Таблицы стилей (css/style.css) и клиентский скрипт (js/app.js)
├── notebooks/              # Исследовательские ноутбуки (01_eda_sms.ipynb ... 05_url_models.ipynb)
├── docs/                   # Отчёты, диаграммы, презентация (SMSGwards_Presentation.pptx / .pdf)
├── tests/                  # Автоматические модульные и интеграционные тесты (pytest)
├── risk_config.yaml        # Веса, пороги и правила расчёта уровня риска
├── requirements.txt        # Зафиксированные версии зависимостей
├── .env.example            # Шаблон переменных окружения
└── manage.py
```

---

## REST API (v1) и документация OpenAPI

Интерактивная документация Swagger UI доступна по адресу: `/api/docs/` (схема OpenAPI: `/api/schema/`).

| Метод | Эндпоинт | Назначение | Коды ответа |
|---|---|---|---|
| `POST` | `/api/v1/auth/register/` | Регистрация нового пользователя | `201`, `400` |
| `POST` | `/api/v1/auth/login/` | Авторизация (блокировка на 60 с после 5 неудач) | `200`, `401`, `429` |
| `POST` | `/api/v1/auth/logout/` | Выход из системы (уничтожение сессии) | `204` |
| `POST` | `/api/v1/check/sms/` | Совместный анализ SMS и вложенных ссылок (до 30/мин) | `200`, `400`, `429`, `503` |
| `POST` | `/api/v1/check/url/` | Структурный анализ отдельного URL (до 30/мин) | `200`, `400`, `429`, `503` |
| `GET` | `/api/v1/checks/` | История проверок с фильтрами и пагинацией (по 20) | `200` |
| `GET` | `/api/v1/checks/<id>/` | Подробный результат проверки (только владелец) | `200`, `404` |
| `DELETE` | `/api/v1/checks/<id>/` | Удаление собственной записи проверки | `204`, `404` |
| `GET` | `/api/v1/stats/summary/` | Сводные данные для Dashboard (`?days=7\|30\|90`) | `200` |
| `GET` | `/api/v1/health/` | Проверка работоспособности сервера и статуса моделей | `200` |

### Примеры запросов (`curl`)

```bash
# Проверка статуса сервиса
curl -s http://127.0.0.1:8000/api/v1/health/

# Проверка SMS-сообщения (с базовой аутентификацией или сессией)
curl -u demo_user:DemoPass123! -X POST http://127.0.0.1:8000/api/v1/check/sms/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Ваша карта заблокирована! Срочно подтвердите данные: http://secure-bank-verify.example/login"}'

# Проверка отдельного URL
curl -u demo_user:DemoPass123! -X POST http://127.0.0.1:8000/api/v1/check/url/ \
  -H "Content-Type: application/json" \
  -d '{"url": "http://xn--exmple-cua.test/secure/login?verify=1"}'
```

---

## Тестирование и переобучение моделей

```bash
# Проверка конфигурации Django
python manage.py check

# Запуск всех тестов с замером покрытия
pytest --cov=ml --cov=apps

# Пересборка мультиязычного датасета (UCI + HuggingFace: ruSpamModels, salyamq/kazspam-v1, kz-transformers) и переобучение SMS-модели
python -c "from ml.train_sms import run_all; run_all(force_reload=True, download_missing=False)"

# Синхронизация таблицы ModelVersion в БД с новыми артефактами
python manage.py sync_model_versions
```

### Актуальные модели (`ml/artifacts/`)
- **SMS-классификатор (`v2.0.0`)**: `Logistic Regression (balanced)` (TF-IDF word + char_wb + 6 ручных признаков, нормализация смешанно-скриптовых гомоглифов, языки: `en`, `ru`, `kk`; тест: `ROC-AUC = 0.9961`, `F1 = 0.9532`, `Recall = 0.9766`).
- **URL-классификатор (`v1.0.0`)**: `HistGradientBoosting` по 22 структурным признакам URL (без сетевых запросов).
