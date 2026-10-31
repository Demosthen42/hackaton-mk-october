"""Сравнение трёх режимов поиска на контрольных вопросах.

    python compare_backends.py            # таблица hit@1 / hit@3
    python compare_backends.py --verbose  # плюс промахи по каждому вопросу

Считается только retrieval: LLM не вызывается, ключ не нужен.
Сам прогон выполняет `evaluation.check_all_questions` — одна логика на все отчёты.
"""
from __future__ import annotations

import argparse
import time

from evaluation import check_all_questions, summarize
from semantic import SemanticUnavailable


def warmup() -> bool:
    """Грузим модель до замеров, иначе холодный старт попадёт в первую строку таблицы."""
    try:
        from semantic import _index

        started = time.perf_counter()
        _index()
    except (ImportError, SemanticUnavailable) as exc:
        print(f"Внимание: {exc}\nСтроки semantic/hybrid будут недоступны.\n")
        return False
    print(f"Индекс и модель загружены за {time.perf_counter() - started:.1f} с\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="показать промахи по вопросам")
    args = parser.parse_args()

    has_semantic = warmup()
    backends = ("lexical", "semantic", "hybrid") if has_semantic else ("lexical",)

    print(f"{'режим':<10} {'hit@1':>7} {'hit@3':>7} {'мс/запрос':>11}   отказы")
    print("-" * 50)

    for backend in backends:
        try:
            results = check_all_questions(backend)
        except Exception as exc:  # 503 без model2vec и прочие причины
            print(f"{backend:<10} недоступен: {exc}")
            continue
        summary = summarize(results)
        total = summary["answerable"]
        print(
            f"{backend:<10} {summary['hit_at_1']:>4}/{total} "
            f"{summary['hit_at_3']:>4}/{total} {summary['p50_latency_ms']:>10.1f}"
            f"   отказы {summary['abstain_correct']}/{summary['abstain_total']}"
        )
        if args.verbose:
            for item in results:
                if item.passed:
                    continue
                found = item.found_sources or ["—"]
                print(f"    промах {item.id}: {item.question}")
                print(f"      ждали {item.expected_sources}, получили {found}")



if __name__ == "__main__":
    main()
