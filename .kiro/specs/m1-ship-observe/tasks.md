# План реализации: M1 — Ship And Observe

## Обзор

Реализация переводит `crop-ml-api` из состояния «работает локально» в состояние «готов к деплою и наблюдаем». Работа разбита на шесть последовательных этапов: зависимости → ядро (логирование + метрики) → middleware → рефакторинг модели → интеграция в main → контейнеризация → тесты → документация.

## Задачи

- [x] 1. Добавить зависимости в `pyproject.toml`
  - Добавить `prometheus-client` и `python-json-logger` в секцию `dependencies` через `uv add prometheus-client python-json-logger`
  - Добавить `hypothesis>=6.100` в `[dependency-groups] dev` через `uv add --dev "hypothesis>=6.100"`
  - Убедиться, что `uv.lock` обновлён после добавления зависимостей
  - _Требования: 3.1, 3.2, 3.3, 3.4_

- [-] 2. Создать модуль структурированного логирования `app/core/`
  - [x] 2.1 Создать `app/core/__init__.py` (пустой файл-маркер пакета)
    - _Требования: 2.1, 2.4_

  - [x] 2.2 Создать `app/core/logging.py` с функциями `configure_logging()` и `get_logger()`
    - Реализовать `configure_logging()`: читает `LOG_LEVEL` из env (fallback `"info"`), настраивает корневой логгер с `JsonFormatter` из `pythonjsonlogger`
    - Форматтер: `fmt="%(asctime)s %(name)s %(levelname)s %(message)s"`, `rename_fields={"levelname": "level", "asctime": "timestamp"}`
    - `root.handlers.clear()` перед добавлением нового хендлера — предотвращает дублирование при повторном вызове
    - Реализовать `get_logger(name: str) -> logging.Logger` как тонкую обёртку над `logging.getLogger`
    - _Требования: 2.1, 2.4, 2.6_

  - [ ]* 2.3 Написать property-тест для `app/core/logging.py`
    - **Свойство 1: Каждая запись лога — валидный JSON**
    - **Validates: Requirements 2.1, 2.4**
    - Использовать `hypothesis`: `@given(message=st.text(min_size=1), extra_fields=st.dictionaries(st.text(min_size=1), st.text()))`, `@settings(max_examples=100)`
    - Перехватывать stdout через `logging.StreamHandler(io.StringIO)`, вызывать `logger.info(message, extra=extra_fields)`, парсить вывод через `json.loads`
    - Тег: `# Feature: m1-ship-observe, Property 1: каждая запись лога — валидный JSON`

- [x] 3. Создать модуль Prometheus-метрик `app/core/metrics.py`
  - Создать явный `CollectorRegistry()` (не глобальный `REGISTRY`) — изолирует метрики для тестирования
  - Определить `http_requests_total` (Counter, метки: `method`, `path`, `status_code`)
  - Определить `http_request_duration_seconds` (Histogram, метки: `method`, `path`)
  - Определить `inference_duration_seconds` (Histogram, без меток)
  - Экспортировать `registry`, `generate_latest`, `CONTENT_TYPE_LATEST` для использования в `main.py`
  - _Требования: 3.1, 3.2, 3.3, 3.4_

- [ ] 4. Создать ASGI-middleware `app/api/middleware.py`
  - [ ] 4.1 Создать `app/api/__init__.py` (пустой файл-маркер пакета)
    - _Требования: 2.2, 2.3, 3.5_

  - [ ] 4.2 Создать `app/api/middleware.py` с классом `ObservabilityMiddleware(BaseHTTPMiddleware)`
    - Метод `dispatch`: генерировать `request_id = str(uuid.uuid4())`, замерять время через `time.perf_counter()`
    - После получения ответа: логировать INFO с полями `request_id`, `method`, `path`, `status_code`, `duration_ms`
    - Инкрементировать `http_requests_total` и записывать в `http_request_duration_seconds` для всех путей кроме `EXCLUDED_PATHS = {"/metrics"}`
    - Добавлять заголовок `X-Request-ID: {request_id}` в ответ
    - При необработанном исключении: логировать ERROR с `request_id`, `exc_type`, `exc_message`, затем пробрасывать исключение дальше
    - _Требования: 2.2, 2.3, 2.5, 3.5, 3.7_

- [ ] 5. Рефакторинг `app/model.py`: убрать `lru_cache`, добавить `load_model()`
  - Удалить декоратор `@lru_cache(maxsize=1)` и функцию `get_model()`
  - Добавить функцию `load_model() -> torch.nn.Module`: проверяет существование `MODEL_PATH`, загружает модель через `torch.load(..., weights_only=False)`, вызывает `model.eval()`, возвращает модель
  - Обновить сигнатуру `predict()`: добавить параметр `model: torch.nn.Module` (явная передача вместо глобального кэша)
  - Внутри `predict()` убрать вызов `get_model()`, использовать переданный `model`
  - _Требования: 1.1, 1.2, 1.3, 1.5_

- [ ] 6. Обновить `app/main.py`: lifespan, middleware, `/metrics`
  - Добавить `asynccontextmanager lifespan(app)`: вызвать `configure_logging()` первым, затем `load_model()` → `app.state.model`; логировать INFO `event="model_loaded"` с путём к файлу весов; `FileNotFoundError` не перехватывать
  - Передать `lifespan=lifespan` в конструктор `FastAPI(...)`
  - Зарегистрировать `app.add_middleware(ObservabilityMiddleware)`
  - Добавить эндпоинт `GET /metrics` (с `include_in_schema=False`): возвращать `PlainTextResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)`
  - Обновить вызовы `predict()` в `predict_endpoint` и `predict_demo`: передавать `model=request.app.state.model`
  - _Требования: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1, 3.6_

- [ ] 7. Контрольная точка — убедиться, что все существующие тесты проходят
  - Запустить `uv run pytest tests/test_api.py tests/test_preprocess.py tests/test_schemas.py -v`
  - Убедиться, что изменения сигнатуры `predict()` и lifespan не сломали существующие тесты
  - Если тесты падают — исправить до перехода к следующему этапу
  - _Требования: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 8. Написать тесты `tests/test_health.py`
  - [ ] 8.1 Написать примерные тесты для health endpoint и X-Request-ID
    - `test_health_returns_ok`: `GET /health` возвращает `{"status": "ok"}` со статусом 200
    - `test_health_has_request_id_header`: ответ содержит заголовок `X-Request-ID`
    - `test_request_id_is_uuid4`: значение `X-Request-ID` — валидный UUID v4 (проверить через `uuid.UUID(value, version=4)`)
    - `test_metrics_endpoint_returns_200`: `GET /metrics` возвращает статус 200
    - `test_metrics_content_type`: Content-Type ответа `/metrics` содержит `text/plain`
    - _Требования: 2.2, 3.1_

  - [ ]* 8.2 Написать property-тест: X-Request-ID присутствует в каждом ответе
    - **Свойство 2: X-Request-ID присутствует в каждом ответе**
    - **Validates: Requirements 2.2**
    - `@given(n=st.integers(min_value=1, max_value=5))`, `@settings(max_examples=100)`
    - Отправлять `POST /predict/demo?n={n}`, проверять наличие и валидность UUID v4 в `X-Request-ID`
    - Тег: `# Feature: m1-ship-observe, Property 2: X-Request-ID присутствует в каждом ответе`

  - [ ]* 8.3 Написать property-тест: лог запроса содержит все обязательные поля
    - **Свойство 3: Лог запроса содержит все обязательные поля**
    - **Validates: Requirements 2.3**
    - `@given(n=st.integers(min_value=1, max_value=5))`, `@settings(max_examples=100)`
    - Перехватывать лог-записи через `caplog` или кастомный хендлер, проверять наличие полей `request_id`, `method`, `path`, `status_code`, `duration_ms`
    - Тег: `# Feature: m1-ship-observe, Property 3: лог запроса содержит все обязательные поля`

- [ ] 9. Написать тесты `tests/test_predict_contract.py`
  - [ ] 9.1 Написать примерные тесты regression для fixture payload
    - `test_fixture_predict_count`: fixture payload (N=10) возвращает ровно 10 predictions
    - `test_fixture_predict_proba_finite`: все `proba` из fixture конечны (`math.isfinite`)
    - `test_predict_422_wrong_timesteps`: features с неверным числом timesteps → 422
    - `test_predict_422_wrong_channels`: features с неверным числом каналов → 422
    - _Требования: 6.1, 6.2, 6.3, 6.5_

  - [ ]* 9.2 Написать property-тест: все вероятности конечны для любого валидного входа
    - **Свойство 6: Prediction contract — все вероятности конечны**
    - **Validates: Requirements 6.1, 6.3, 6.4**
    - `@given(n=st.integers(min_value=1, max_value=4), nan_fraction=st.floats(min_value=0.0, max_value=0.5))`, `@settings(max_examples=100)`
    - Генерировать features с частичными NaN, отправлять `POST /predict`, проверять `math.isfinite(p["proba"])` для всех predictions
    - Тег: `# Feature: m1-ship-observe, Property 6: все proba конечны`

  - [ ]* 9.3 Написать property-тест: невалидная форма features возвращает 422
    - **Свойство 7: Невалидная форма features возвращает 422**
    - **Validates: Requirements 6.5**
    - `@given(wrong_t=st.integers(min_value=1, max_value=50).filter(lambda x: x != 26))`, `@settings(max_examples=100)`
    - Генерировать features с неверным числом timesteps, проверять статус 422
    - Тег: `# Feature: m1-ship-observe, Property 7: невалидная форма → 422`

  - [ ]* 9.4 Написать property-тесты: метрики HTTP инкрементируются и инференс записывает наблюдение
    - **Свойство 4: Метрики HTTP инкрементируются после каждого запроса**
    - **Свойство 5: Инференс записывает наблюдение в гистограмму**
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6**
    - `@given(n=st.integers(min_value=1, max_value=3))`, `@settings(max_examples=50)`
    - Считывать значения счётчика до и после запроса через `app/core/metrics.py` registry, проверять инкремент `http_requests_total` и наличие нового наблюдения в `inference_duration_seconds`
    - Тег: `# Feature: m1-ship-observe, Property 4+5: метрики инкрементируются`

- [ ] 10. Контрольная точка — убедиться, что все тесты проходят
  - Запустить `uv run pytest tests/ -v`
  - Убедиться, что все новые и существующие тесты проходят
  - Если тесты падают — исправить до перехода к следующему этапу

- [ ] 11. Создать `Dockerfile`
  - Базовый образ: `python:3.11-slim`
  - Установить `uv` через `COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv`
  - `WORKDIR /app`
  - Скопировать `pyproject.toml` и `uv.lock`, выполнить `RUN uv sync --frozen --no-dev` (зависимости кэшируются отдельным слоем)
  - Скопировать `app/`, `src/`, `models/`
  - Создать системного пользователя: `RUN adduser --system --no-create-home appuser` и `USER appuser`
  - `EXPOSE 8000`
  - `CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`
  - _Требования: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 12. Создать `docker-compose.yml`
  - Определить сервис `api` с `build: .`
  - Пробросить порт `"8000:8000"`
  - Монтировать `./models:/app/models:ro` как volume (веса не включаются в образ при локальной разработке)
  - Передавать `LOG_LEVEL: ${LOG_LEVEL:-info}` через `environment`
  - Добавить `healthcheck`: `test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]`, `interval: 10s`, `timeout: 5s`, `retries: 3`
  - _Требования: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 13. Обновить `README.md`
  - Добавить раздел с командами запуска и проверки
  - Команда запуска через uv: `uv run uvicorn app.main:app --reload --port 8000`
  - Команда проверки health: `curl -sf http://localhost:8000/health`
  - Команда проверки fixture contract с ожидаемым результатом 10 predictions
  - Команда запуска через Docker Compose: `docker compose up --build`
  - Команда проверки метрик: `curl -s http://localhost:8000/metrics | head -20`
  - _Требования: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 14. Финальная контрольная точка — убедиться, что все тесты проходят
  - Запустить `uv run pytest tests/ -v`
  - Убедиться, что все тесты проходят без ошибок

## Примечания

- Задачи, помеченные `*`, являются опциональными и могут быть пропущены для ускоренного MVP
- Каждая задача ссылается на конкретные требования для трассируемости
- Контрольные точки (задачи 7, 10, 14) обеспечивают инкрементальную валидацию
- Property-тесты проверяют универсальные свойства корректности (Свойства 1–7 из design.md)
- Примерные тесты проверяют конкретные сценарии и граничные случаи
- Изменение сигнатуры `predict()` (добавление параметра `model`) не затрагивает существующие тесты — они используют HTTP-клиент
