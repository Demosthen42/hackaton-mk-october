import importlib.util
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from retrieval import retrieve
from semantic import SemanticUnavailable, embed_model, hybrid_retrieve, semantic_retrieve

load_dotenv()
app = FastAPI(title="Hackathon AI Assistant", version="0.2.0")

RetrievalBackend = Literal["lexical", "semantic", "hybrid"]

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    retrieval: RetrievalBackend = Field(
        default="lexical",
        description=(
            "lexical — поиск из retrieval.py (с намеренными недочётами, режим по умолчанию); "
            "semantic — только эмбеддинги; hybrid — лексика + эмбеддинги из semantic.py"
        ),
    )

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    mode: str
    retrieval: str
    retrieved: list[dict[str, Any]]

def get_contexts(question: str, backend: str) -> list[dict[str, Any]]:
    if backend == "lexical":
        return retrieve(question)
    try:
        if backend == "semantic":
            return semantic_retrieve(question)
        return hybrid_retrieve(question)
    except SemanticUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

def make_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[Источник: {item['source']}]\n{item['text']}" for item in contexts
    )
    return f"""Ты помощник участника хакатона. Отвечай по-русски кратко.
Используй только контекст. Если ответа нет в контексте, честно скажи, что не нашёл информации.
В конце укажи источники в формате «Источники: имя_файла».
Вопрос: {question}
Контекст:
{context_text}
"""

def llm_answer(question: str, contexts: list[dict]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        kwargs = {"api_key": api_key}
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": "Отвечай только по переданным источникам. Не выдумывай факты."},
                {"role": "user", "content": make_prompt(question, contexts)},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        # Учебный проект: показываем понятную ошибку, не падаем всем API.
        return f"Ошибка вызова LLM ({type(exc).__name__} {exc}). Проверьте ключ, модель и доступность API."

def demo_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "Не нашёл подтверждённой информации в документах."
    # Детерминированная заглушка, чтобы сервис работал без API-ключа.
    first_line = next(
        (line.strip("- ").strip() for line in contexts[0]["text"].splitlines()
         if line.strip() and not line.startswith("#")),
        contexts[0]["text"][:180]
    )
    return f"Демо-режим: наиболее близкий фрагмент — «{first_line}»."

@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "model": os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        "semantic_available": importlib.util.find_spec("model2vec") is not None,
        "embed_model": embed_model(),
        "retrieval_backends": ["lexical", "semantic", "hybrid"],
    }

@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    contexts = get_contexts(payload.question, payload.retrieval)
    generated = llm_answer(payload.question, contexts)
    if generated is None:
        answer = demo_answer(payload.question, contexts)
        mode = "demo"
    else:
        answer = generated
        mode = "llm"
    sources = [item["source"] for item in contexts]
    return {
        "answer": answer,
        "sources": sources,
        "mode": mode,
        "retrieval": payload.retrieval,
        "retrieved": [
            {"source": item["source"], "score": round(item["score"], 3),
             "matched_tokens": item["matched_tokens"]}
            for item in contexts
        ],
    }
