"""Проверка всех контрольных вопросов из `evals.json`.

Главная точка входа — `check_all_questions()`: прогоняет весь набор через тот же
код, что и `POST /ask`, и возвращает результат по каждому вопросу.

    from evaluation import check_all_questions, summarize

    results = check_all_questions("hybrid")
    print(summarize(results)["hit_at_1"], "из", len(results))

CLI:

    python evaluation.py                    # режим lexical, без LLM (быстро, офлайн)
    python evaluation.py --backend hybrid --verbose
    python evaluation.py --llm              # честная проверка ответов, 18 вызовов LLM
    python evaluation.py --json baseline.json

Без флага `--llm` сервис принудительно работает в demo-режиме: результат
детерминирован, ключ и сеть не нужны. С флагом `--llm` ответы проверяются
по факту (поле `must_contain` в `evals.json`, если оно задано).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).parent
EVALS_PATH = ROOT / "evals.json"
BACKENDS = ("lexical", "semantic", "hybrid")


@dataclass
class QuestionResult:
    """Результат одного вопроса: что нашли, что ждали и сколько это заняло."""

    id: str
    question: str
    expected_sources: list[str]
    found_sources: list[str] = field(default_factory=list)
    hit_at_1: bool = False
    hit_at_3: bool = False
    answer: str | None = None
    mode: str | None = None
    latency_ms: float = 0.0
    missing_facts: list[str] = field(default_factory=list)
    error: str | None = None
    expect_abstain: bool = False
    abstained: bool = False

    @property
    def passed(self) -> bool:
        """Прошёл — если нужный источник первый, а для вопроса без ответа — пустая выдача."""
        if self.error is not None:
            return False
        return self.abstained if self.expect_abstain else self.hit_at_1


def load_evals(path: Path | str = EVALS_PATH) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextlib.contextmanager
def _without_api_key():
    """Временно убираем ключ: так проверка не зависит от сети и не тратит деньги."""
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved


def _check_one(case: dict[str, Any], backend: str) -> QuestionResult:
    from app import AskRequest, ask

    expected = list(case.get("sources", []))
    result = QuestionResult(
        id=str(case.get("id", "")),
        question=str(case["question"]),
        expected_sources=expected,
        expect_abstain=bool(case.get("expect_abstain", False)),
    )

    started = time.perf_counter()
    try:
        body = ask(AskRequest(question=result.question, retrieval=backend))
    except Exception as exc:  # 503 без model2vec, битый evals.json и т.п.
        result.latency_ms = (time.perf_counter() - started) * 1000
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.latency_ms = (time.perf_counter() - started) * 1000
    result.found_sources = list(body.get("sources", []))
    result.answer = body.get("answer")
    result.mode = body.get("mode")

    top = result.found_sources[:1]
    result.hit_at_1 = bool(set(top) & set(expected))
    result.hit_at_3 = bool(set(result.found_sources[:3]) & set(expected))
    # Для вопроса без ответа «прошёл» означает, что поиск ничего не выдумал.
    result.abstained = not result.found_sources

    # Факты проверяем только у ответов модели: demo-ответ — это фрагмент документа,
    # и проверка по нему говорит о демо-заглушке, а не о качестве ответа.
    must_contain = [str(item) for item in case.get("must_contain", [])]
    if must_contain and result.answer and result.mode == "llm":
        lowered = result.answer.lower()
        result.missing_facts = [fact for fact in must_contain if fact.lower() not in lowered]

    # Намеренный недочёт №? в app.py: ошибка провайдера приходит как обычный ответ.
    if result.mode == "llm" and (result.answer or "").startswith("Ошибка вызова LLM"):
        result.error = result.answer

    return result


def check_all_questions(
    backend: str = "lexical",
    *,
    use_llm: bool = False,
    cases: Iterable[dict[str, Any]] | None = None,
) -> list[QuestionResult]:
    """Проверяет все вопросы и возвращает по одному результату на вопрос.

    backend: "lexical" (стартовый код с недочётами), "semantic" или "hybrid".
    use_llm: False — принудительный demo-режим (быстро и воспроизводимо).
    cases: свой набор вопросов того же формата, что и evals.json.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Неизвестный режим поиска: {backend!r}. Доступны: {', '.join(BACKENDS)}")

    questions = list(cases) if cases is not None else load_evals()
    if not questions:
        return []

    # Импортируем app ДО снятия ключа: при импорте выполняется load_dotenv(),
    # и он вернул бы ключ из .env обратно.
    from app import AskRequest, ask  # noqa: F401

    if use_llm:
        return [_check_one(case, backend) for case in questions]
    with _without_api_key():
        if os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Не удалось отключить LLM: ключ остался в окружении")
        return [_check_one(case, backend) for case in questions]


def summarize(results: Iterable[QuestionResult]) -> dict[str, Any]:
    """Сводка: hit@k по вопросам с ответом, отказы по вопросам без ответа, ошибки, время."""
    items = list(results)
    answerable = [item for item in items if not item.expect_abstain]
    abstain = [item for item in items if item.expect_abstain]
    hit1 = sum(item.hit_at_1 for item in answerable)
    hit3 = sum(item.hit_at_3 for item in answerable)
    latencies = sorted(item.latency_ms for item in items)
    return {
        "total": len(items),
        "answerable": len(answerable),
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_1_rate": round(hit1 / len(answerable), 3) if answerable else 0.0,
        "hit_at_3_rate": round(hit3 / len(answerable), 3) if answerable else 0.0,
        "abstain_total": len(abstain),
        "abstain_correct": sum(item.abstained for item in abstain),
        "empty_retrieval": sum(1 for item in items if not item.found_sources),
        "errors": sum(1 for item in items if item.error),
        "wrong_facts": sum(1 for item in items if item.missing_facts),
        "failed_ids": [item.id for item in items if not item.passed],
        "modes": dict(Counter(item.mode for item in items if item.mode)),
        "p50_latency_ms": round(latencies[len(latencies) // 2], 1) if latencies else 0.0,
    }


def format_table(results: Iterable[QuestionResult]) -> str:
    lines = [f"{'ID':<5}{'рез':<5}{'нашли':<32}{'ждали':<30}{'мс':>7}"]
    lines.append("-" * 79)
    for item in results:
        found = item.found_sources[0] if item.found_sources else "—"
        if item.error:
            verdict = "ERR"
        else:
            verdict = "✓" if item.passed else "✗"
        lines.append(
            f"{item.id:<5}{verdict:<5}{found:<32}{', '.join(item.expected_sources):<30}{item.latency_ms:>7.1f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка всех вопросов из evals.json")
    parser.add_argument("--backend", choices=BACKENDS, default="lexical")
    parser.add_argument("--llm", action="store_true", help="проверять ответы через LLM (медленно, платно)")
    parser.add_argument("--verbose", action="store_true", help="показать таблицу по всем вопросам")
    parser.add_argument("--json", dest="json_path", help="сохранить результаты в JSON")
    args = parser.parse_args()

    if args.llm:
        print("Флаг --llm: будет столько вызовов провайдера, сколько вопросов.\n")

    results = check_all_questions(args.backend, use_llm=args.llm)
    summary = summarize(results)

    if args.verbose or args.llm:
        print(format_table(results))
        print()

    modes = ", ".join(f"{name}: {count}" for name, count in summary["modes"].items()) or "—"
    print(f"Режим поиска: {args.backend}; фактический режим ответов: {modes}")
    print(
        f"hit@1 {summary['hit_at_1']}/{summary['answerable']}, "
        f"hit@3 {summary['hit_at_3']}/{summary['answerable']}, "
        f"отказы {summary['abstain_correct']}/{summary['abstain_total']}, "
        f"пустых находок {summary['empty_retrieval']}, "
        f"ошибок {summary['errors']}, "
        f"p50 {summary['p50_latency_ms']} мс"
    )

    if summary["failed_ids"]:
        print("Не прошли:", ", ".join(summary["failed_ids"]))
    if summary["wrong_facts"]:
        details = [f"{item.id}: нет {item.missing_facts}" for item in results if item.missing_facts]
        print("Ответ без нужных фактов: " + "; ".join(details))

    if args.json_path:
        payload = {"summary": summary, "results": [asdict(item) for item in results]}
        Path(args.json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Сохранено: {args.json_path}")

    return 1 if summary["failed_ids"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
