import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("OPENAI_API_KEY", "").strip()
if not key:
    raise SystemExit("OPENAI_API_KEY не задан. Заполните .env и повторите запуск.")

from openai import OpenAI
kwargs = {"api_key": key}
base_url = os.getenv("OPENAI_BASE_URL", "").strip()
if base_url:
    kwargs["base_url"] = base_url
client = OpenAI(**kwargs)
model = os.getenv("LLM_MODEL", "")
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Ответь ровно одним словом: готово"}],
    temperature=0,
)
print("LLM доступна. Модель:", model)
print("Ответ:", response.choices[0].message.content)
