"""

Зависимость только для semantic/hybrid:
    pip install -r requirements-semantic.txt

Модель по умолчанию ~530 МБ, скачивается один раз в кэш Hugging Face.
Для офлайна: сохраните модель локально и укажите путь в EMBED_MODEL.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from retrieval import STOPWORDS, load_documents, retrieve

DEFAULT_EMBED_MODEL = "minishlab/potion-multilingual-128M"
TOP_K = 1


def embed_model() -> str:
    """Читаем env лениво: .env загружается уже после импорта этого модуля."""
    return os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)


class SemanticUnavailable(RuntimeError):
    """Семантический поиск нельзя использовать: нет зависимости или модели."""


def chunk_document(doc: dict) -> list[dict]:
    """Режем документ на абзацы и добавляем заголовок: он даёт чанку тему."""
    lines = doc["text"][:320].splitlines()
    title = next((line.lstrip("# ").strip() for line in lines if line.startswith("#")), "")
    chunks = []
    for line in lines:
        text = line.lstrip("- ").strip()
        if text and not text.startswith("#"):
            chunks.append(
                {
                    "source": doc["source"],
                    "text": text,
                    "embed_text": f"{title}. {text}",
                    "matched_tokens": [],
                }
            )
    return chunks


@lru_cache(maxsize=1)
def _chunks() -> list[dict[str, Any]]:
    """Чанки без модели: лексической половине эмбеддинги не нужны."""
    return [chunk for doc in load_documents() for chunk in chunk_document(doc)]


@lru_cache(maxsize=1)
def _index():
    """Эмбеддинги считаются один раз на процесс. Первый вызов ~5 секунд."""
    try:
        from model2vec import StaticModel
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise SemanticUnavailable(
            "Семантический поиск недоступен: установите model2vec "
            "(pip install -r requirements-semantic.txt)."
        ) from exc

    import numpy as np

    model_name = embed_model()
    try:
        model = StaticModel.from_pretrained(model_name)
    except Exception as exc:  # нет сети, битый кэш, опечатка в EMBED_MODEL
        raise SemanticUnavailable(
            f"Не удалось загрузить модель эмбеддингов «{model_name}»: {exc}"
        ) from exc

    chunks = _chunks()
    vectors = model.encode([chunk["embed_text"] for chunk in chunks])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return model, chunks, vectors


def semantic_retrieve(
    question: str, top_k: int = TOP_K, min_score: float = 0.0, dedupe: bool = True
) -> list[dict[str, Any]]:
    # Порог по косинусу по умолчанию выключен: его нужно калибровать
    # на вопросах без ответа, иначе он режет верные находки.
    import numpy as np

    model, chunks, vectors = _index()
    query = model.encode([question])[0]
    query = query / np.linalg.norm(query)
    scores = vectors @ query

    result, seen = [], set()
    # Берём с запасом: без запаса после дедупликации можно остаться с одним источником.
    limit = top_k * 4 if dedupe else top_k
    for i in np.argsort(-scores)[:limit]:
        item = chunks[int(i)]
        if float(scores[i]) < min_score:
            continue
        if dedupe and item["source"] in seen:
            continue
        seen.add(item["source"])
        result.append({**item, "score": float(scores[i])})
        if len(result) == top_k:
            break
    return result


def _tokens(text: str) -> list[str]:
    """Токены со «учебным» стеммингом: первые 5 символов снимают морфологию."""
    return [
        token[:5]
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9#-]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def bm25_retrieve(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Полный текст документов, BM25 (k1=1.5, b=0.75), без обрезки в 260 символов."""
    chunks = _chunks()
    chunk_tokens = [_tokens(chunk["embed_text"]) for chunk in chunks]
    df = Counter(token for tokens in chunk_tokens for token in set(tokens))
    avg_len = sum(len(tokens) for tokens in chunk_tokens) / len(chunk_tokens)
    query_tokens = _tokens(question)

    scored = []
    for chunk, tokens in zip(chunks, chunk_tokens):
        tf = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            if token in tf:
                idf = math.log(1 + (len(chunks) - df[token] + 0.5) / (df[token] + 0.5))
                score += idf * tf[token] * 2.5 / (
                    tf[token] + 1.5 * (0.25 + 0.75 * len(tokens) / avg_len)
                )
        if score > 0:
            scored.append(
                {
                    **chunk,
                    "score": score,
                    "matched_tokens": sorted(set(query_tokens) & set(tokens)),
                }
            )
    return sorted(scored, key=lambda item: -item["score"])[:top_k]


def hybrid_retrieve(question: str, top_k: int = TOP_K, c: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion: важна позиция в списке, а не абсолютный скор."""
    fused: dict[tuple[str, str], dict[str, Any]] = {}
    semantic = semantic_retrieve(question, 2, dedupe=False)
    for ranked in (bm25_retrieve(question, 3), semantic):
        for rank, item in enumerate(ranked):
            key = (item["source"], item["text"])
            fused.setdefault(key, {**item, "score": 0.0})
            fused[key]["score"] += 1 / (c + rank)

    result, seen = [], set()
    for item in sorted(fused.values(), key=lambda entry: -entry["score"]):
        if item["source"] in seen:  # один источник в ответе — один раз
            continue
        seen.add(item["source"])
        result.append(item)
        if len(result) == top_k:
            break
    return result
