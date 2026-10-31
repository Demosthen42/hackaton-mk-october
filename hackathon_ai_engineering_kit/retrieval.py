"""Учебный лексический поиск. В файле намеренно оставлены ошибки для упражнения."""
from pathlib import Path
import re

DATA_DIR = Path(__file__).parent / "data"
STOPWORDS = {"когда", "где", "какой", "какая", "какие", "можно", "ли", "это", "для", "что", "и", "в", "на", "по", "с", "из", "от", "до"}

def tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9#-]+", text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    }

def load_documents() -> list[dict]:
    docs = []
    for path in sorted(DATA_DIR.glob("*.md")):
        docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs

def retrieve(question: str, top_k: int = 1) -> list[dict]:
    """Возвращает top_k фрагментов с лексическим пересечением."""
    q_tokens = tokenize(question)
    scored = []
    for doc in load_documents():
        indexed_text = doc["text"][:260]
        overlap = q_tokens & tokenize(indexed_text)
        score = len(overlap) / max(len(q_tokens), 1)
        if score > 0:
            scored.append({**doc, "score": score, "matched_tokens": sorted(overlap)})
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]
