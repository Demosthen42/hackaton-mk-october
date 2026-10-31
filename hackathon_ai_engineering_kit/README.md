# AI Engineering Hackathon Workshop Kit

Практический набор для мастер-класса «От LLM к AI-продукту за 90 минут».

## Что внутри
- `data/` — 9 документов реального хакатона: регламент «Код.Город.Будущее» (31.10–01.11.2026) и кейс Ujin OS.
- `evals.json` — 25 контрольных вопросов с эталонными ответами и 4 вопроса без ответа (проверка отказов).
- `app.py` — стартовый FastAPI-сервис.
- `retrieval.py` — простой лексический поиск с намеренно оставленными недочётами.
- `semantic.py` — эталон: семантический поиск на эмбеддингах и гибридный через RRF (без недочётов).
- `compare_backends.py` — сравнение `lexical` / `semantic` / `hybrid` на `evals.json`.
- `evaluation.py` — `check_all_questions()`: прогон всех вопросов через `/ask` с отчётом по каждому и сводкой.
- `tasks.md` — базовое и продвинутое задания.
- `tests/` — smoke-тесты API, тесты режимов поиска и `conftest.py` с офлайн-режимом.
- `.env.example` — пример настройки LLM и модели эмбеддингов.

Документы в `data/` составлены по официальному регламенту хакатона «Код.Город.Будущее» (31 октября — 1 ноября 2026, Пермь) и по описанию кейса Ujin OS; вопросы в `evals.json` проверяются по этим же файлам.
- `requirements.txt` — зависимости; `requirements-semantic.txt` — опционально для семантики.

## Быстрый запуск (Python 3.11+)

### Windows PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-semantic.txt
Copy-Item .env.example .env
uvicorn app:app --reload
pytest -q
python evaluation.py --verbose
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-semantic.txt
cp .env.example .env
uvicorn app:app --reload
pytest -q
python evaluation.py --verbose
```

Откройте `http://127.0.0.1:8000/docs`. Без API-ключа сервис запускается в `demo`-режиме, чтобы упражнение не зависело от интернета и биллинга. В этом режиме ответ формируется детерминированно на основе найденного фрагмента; это демонстрационная заглушка, а не полноценная генерация LLM.

## Подключение LLM
1. Получите API-ключ у выбранного провайдера, разрешённого организаторами.
2. Запишите его в `.env` в `OPENAI_API_KEY`.
3. При необходимости задайте `OPENAI_BASE_URL` для совместимого API и `LLM_MODEL` для доступной модели.
4. Перезапустите приложение и вызовите `GET /health`.

Пример конфигурации:
```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=
LLM_MODEL=gpt-4.1-mini
```
Код использует совместимый с OpenAI Chat Completions API. Поддержка `base_url` зависит от выбранного провайдера. Не коммитьте `.env` и никогда не вставляйте ключ в исходный код или презентацию.

**Важно:** в этом наборе нет настоящего API-ключа, а доступ к LLM не может считаться проверенным до запуска `python check_llm.py` с вашими учётными данными и успешного ответа. Для мероприятия заранее проверьте ключ, модель, лимиты, оплату и доступность сети на всех ноутбуках либо используйте один общий разрешённый организаторами шлюз.

## Семантический и гибридный поиск (опционально)
Базовый набор работает без этой части: `retrieval` по умолчанию — `lexical`.
Чтобы включить эмбеддинги:
```bash
pip install -r requirements-semantic.txt
python compare_backends.py
```
Модель `minishlab/potion-multilingual-128M` (~530 МБ) скачивается один раз в кэш Hugging Face при первом запросе `semantic`/`hybrid`.
Без `model2vec` эти режимы отвечают `503` с подсказкой, лексический поиск продолжает работать.

Для площадки без интернета сохраните модель заранее и укажите путь в `.env`:
```python
from model2vec import StaticModel
StaticModel.from_pretrained("minishlab/potion-multilingual-128M").save_pretrained("models/potion")
```
```env
EMBED_MODEL=models/potion
```

## Проверки
```bash
pytest -q
python check_llm.py
python evaluation.py --verbose
python compare_backends.py
```
`evaluation.py` проверяет все вопросы и печатает таблицу «нашли / ждали», сводку `hit@1`, `hit@3`, корректные отказы, число пустых находок и медиану времени.
По умолчанию ответы берутся из demo-режима (офлайн); `--llm` включает реальные вызовы провайдера, `--json before.json` сохраняет результат для сравнения «до/после».
Код возврата — `1`, если хотя бы один вопрос не прошёл, так что команду можно ставить в CI.
`check_llm.py` проверяет подключение к модели, `compare_backends.py` — качество поиска по `evals.json` (LLM не вызывает).
Тесты работают офлайн: `tests/conftest.py` убирает ключ, поэтому проверяются контракт API и поиск, а не провайдер.

## API
`POST /ask`
```json
{"question": "Во сколько стоп-кодинг?", "retrieval": "lexical"}
```
- `question` — 3–1000 символов.
- `retrieval` — `lexical` (по умолчанию, стартовый `retrieval.py` с недочётами), `semantic` или `hybrid` (эталон из `semantic.py`).
  Неизвестное значение — `422`; `semantic`/`hybrid` без `model2vec` — `503` с подсказкой, что установить.

Ответ содержит `answer`, `sources`, `mode`, `retrieval` и `retrieved`. Источники — имена файлов в `data/`.
Поле `score` в `retrieved` не сравнимо между режимами: у `semantic` это косинус, у `hybrid` — RRF-сумма;
у `lexical` доля совпавших токенов запроса.

`GET /health` дополнительно сообщает `semantic_available`, `embed_model` и список `retrieval_backends`.

## Ограничения учебного проекта
Это учебный MVP: нет аутентификации, rate limit, полноценного ранжирования, логирования пользовательских данных и production hardening. Не размещайте в нём реальные персональные или конфиденциальные данные.
