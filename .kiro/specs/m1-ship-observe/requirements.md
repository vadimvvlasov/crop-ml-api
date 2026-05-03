# Требования: M1 — Ship And Observe

## Введение

Майлстоун M1 переводит существующий FastAPI ML-сервис (`crop-ml-api`) из состояния «работает локально» в состояние «готов к деплою и наблюдаем». Сервис предсказывает класс сельскохозяйственной культуры (Soybean / Corn / Rice / Other) по временным рядам HLS-рефлектанса.

Цель: добавить надёжный старт (lifespan warmup), структурированные JSON-логи с request id, Prometheus-метрики, Docker-образ и docker-compose — без изменения prediction contract (`features (N,26,15)`).

## Глоссарий

- **API** — FastAPI-приложение (`app/main.py`), обслуживающее HTTP-запросы.
- **Model** — PyTorch TL-Transformer, загружаемый из `models/*.pt` через `torch.load`.
- **Lifespan** — механизм FastAPI (`@asynccontextmanager lifespan`) для выполнения кода при старте и остановке приложения.
- **Logger** — компонент структурированного логирования (`app/core/logging.py`), выводящий JSON-строки в stdout.
- **Middleware** — ASGI-middleware (`app/api/middleware.py`), перехватывающий каждый HTTP-запрос для логирования и сбора метрик.
- **MetricsRegistry** — Prometheus-реестр метрик (`app/core/metrics.py`).
- **RequestId** — UUID v4, генерируемый Middleware для каждого входящего запроса.
- **Dockerfile** — инструкция сборки Docker-образа сервиса.
- **Compose** — файл `docker-compose.yml` для локального запуска сервиса.
- **PredictRequest** — Pydantic-схема входного запроса: `features (N,26,15)`, `week_of_year (26,)`, `location (N,2)`.
- **PredictResponse** — Pydantic-схема ответа: список `Prediction(class_id, class_name, proba)`.
- **FixturePayload** — тестовый JSON-файл `fixtures/rs_hls_predict_request.json` с 10 полями.

---

## Требования

### Требование 1: Прогрев модели при старте (Lifespan Warmup)

**User Story:** Как оператор, я хочу, чтобы модель загружалась при старте сервиса, а не на первом запросе, чтобы ошибки загрузки обнаруживались немедленно и первый запрос не был медленнее остальных.

#### Критерии приёмки

1. WHEN приложение запускается, THE API SHALL загрузить Model через FastAPI lifespan до начала обработки запросов.
2. IF файл весов Model не найден при старте, THEN THE API SHALL завершить процесс с ненулевым кодом выхода.
3. THE API SHALL хранить загруженную Model в состоянии приложения (app.state), а не в глобальном `lru_cache`.
4. WHEN Model успешно загружена при старте, THE Logger SHALL записать JSON-сообщение уровня INFO с полем `event: "model_loaded"` и путём к файлу весов.
5. WHEN приложение получает запрос `POST /predict`, THE API SHALL использовать Model из app.state без повторной загрузки.

---

### Требование 2: Структурированные JSON-логи с Request ID

**User Story:** Как разработчик, я хочу видеть структурированные JSON-логи с уникальным идентификатором каждого запроса, чтобы отслеживать запросы в системе и диагностировать проблемы.

#### Критерии приёмки

1. THE Logger SHALL выводить каждую запись лога в формате JSON-строки в stdout.
2. WHEN Middleware получает HTTP-запрос, THE Middleware SHALL сгенерировать RequestId (UUID v4) и добавить его в заголовок ответа `X-Request-ID`.
3. WHEN Middleware обрабатывает запрос, THE Logger SHALL записать JSON-сообщение с полями: `request_id`, `method`, `path`, `status_code`, `duration_ms`.
4. THE Logger SHALL включать поле `level` (`"info"`, `"warning"`, `"error"`) в каждую запись лога.
5. IF при обработке запроса возникает необработанное исключение, THEN THE Logger SHALL записать JSON-сообщение уровня ERROR с полями `request_id`, `exc_type`, `exc_message`.
6. WHERE переменная окружения `LOG_LEVEL` задана, THE Logger SHALL использовать её значение как минимальный уровень логирования; иначе THE Logger SHALL использовать уровень `"info"`.

---

### Требование 3: Prometheus-метрики (`/metrics`)

**User Story:** Как оператор, я хочу получать метрики сервиса в формате Prometheus, чтобы настраивать мониторинг и алерты.

#### Критерии приёмки

1. THE API SHALL предоставлять эндпоинт `GET /metrics`, возвращающий метрики в Prometheus text format (Content-Type: `text/plain; version=0.0.4`).
2. THE MetricsRegistry SHALL экспортировать счётчик `http_requests_total` с метками `method`, `path`, `status_code`.
3. THE MetricsRegistry SHALL экспортировать гистограмму `http_request_duration_seconds` с метками `method`, `path`.
4. THE MetricsRegistry SHALL экспортировать гистограмму `inference_duration_seconds` для измерения времени выполнения `predict()`.
5. WHEN Middleware завершает обработку запроса, THE Middleware SHALL инкрементировать `http_requests_total` и записать наблюдение в `http_request_duration_seconds`.
6. WHEN API выполняет `predict()`, THE API SHALL записать наблюдение в `inference_duration_seconds`.
7. THE API SHALL НЕ включать эндпоинт `/metrics` в подсчёт метрик `http_requests_total` и `http_request_duration_seconds`.

---

### Требование 4: Dockerfile (non-root, production-ready)

**User Story:** Как DevOps-инженер, я хочу собрать Docker-образ сервиса, который запускается от непривилегированного пользователя, чтобы соответствовать требованиям безопасности.

#### Критерии приёмки

1. THE Dockerfile SHALL использовать многоэтапную сборку или минимальный базовый образ Python для уменьшения размера итогового образа.
2. THE Dockerfile SHALL создавать системного пользователя без прав root и запускать процесс uvicorn от его имени.
3. THE Dockerfile SHALL копировать файл весов Model из директории `models/` в образ.
4. THE Dockerfile SHALL устанавливать зависимости через `uv` с использованием CPU-only torch index из `pyproject.toml`.
5. WHEN Docker-контейнер запускается, THE API SHALL быть доступен на порту `8000`.
6. IF файл весов Model отсутствует в образе, THEN THE API SHALL завершить процесс с ненулевым кодом выхода при старте (согласно Требованию 1.2).

---

### Требование 5: Docker Compose для локального запуска

**User Story:** Как разработчик, я хочу запустить сервис одной командой через docker compose, чтобы воспроизвести production-окружение локально.

#### Критерии приёмки

1. THE Compose SHALL определять сервис `api`, собираемый из `Dockerfile` в корне репозитория.
2. WHEN выполняется `docker compose up --build`, THE Compose SHALL запустить API и сделать его доступным на `http://localhost:8000`.
3. THE Compose SHALL монтировать директорию `models/` как volume, чтобы не включать веса в образ при локальной разработке.
4. THE Compose SHALL передавать переменную окружения `LOG_LEVEL` в контейнер (со значением по умолчанию `info`).
5. WHEN API в контейнере запущен, THE API SHALL отвечать `{"status": "ok"}` на `GET /health`.

---

### Требование 6: Сохранение prediction contract

**User Story:** Как потребитель API, я хочу, чтобы эндпоинт `POST /predict` продолжал работать с теми же входными данными и возвращать те же результаты, чтобы изменения M1 не сломали существующих клиентов.

#### Критерии приёмки

1. THE API SHALL принимать `POST /predict` с телом PredictRequest (`features (N,26,15)`, `week_of_year (26,)`, `location (N,2)`) и возвращать PredictResponse.
2. WHEN FixturePayload отправляется в `POST /predict`, THE API SHALL вернуть PredictResponse с ровно 10 элементами в `predictions`.
3. FOR ALL элементов `p` в `predictions` из FixturePayload: значение `p.proba` SHALL быть конечным числом (не NaN, не Inf).
4. THE API SHALL применять интерполяцию NaN перед нормализацией на `NORMALIZATION_FACTOR = 0.7e-4` (порядок операций не меняется).
5. WHEN `POST /predict` получает тело с `features` формы, отличной от `(N,26,15)`, THE API SHALL вернуть HTTP 422.

---

### Требование 7: Обновление README

**User Story:** Как новый разработчик, я хочу найти в README точные команды для запуска и проверки сервиса, чтобы начать работу без дополнительных вопросов.

#### Критерии приёмки

1. THE README SHALL содержать раздел с командой запуска через uv: `uv run uvicorn app.main:app --reload --port 8000`.
2. THE README SHALL содержать команду проверки health: `curl -sf http://localhost:8000/health`.
3. THE README SHALL содержать команду проверки fixture contract с ожидаемым результатом `10`.
4. THE README SHALL содержать команду запуска через Docker Compose: `docker compose up --build`.
5. THE README SHALL содержать команду проверки метрик: `curl -s http://localhost:8000/metrics | head -20`.
