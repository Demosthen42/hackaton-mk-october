import pytest

from evaluation import QuestionResult, check_all_questions, load_evals, summarize

def test_evals_have_required_fields():
    cases = load_evals()
    assert len(cases) >= 20
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["question"]
        if case.get("expect_abstain"):
            assert case["sources"] == []
        else:
            assert case["sources"]

def test_check_all_runs_offline_in_demo(client):
    results = check_all_questions("lexical", cases=load_evals()[:3])
    assert len(results) == 3
    assert {item.mode for item in results} == {"demo"}
    assert all(item.answer for item in results)
    assert all(item.error is None for item in results)

def test_known_question_is_found(client):
    cases = [{"id": "Q11", "question": "Сколько минут отводится на защиту решения?", "sources": ["07_presentation.md"]}]
    item = check_all_questions("lexical", cases=cases)[0]
    assert item.hit_at_1 is True
    assert item.found_sources[0] == "07_presentation.md"

def test_question_without_answer_is_not_found(client):
    cases = [{
        "id": "N01",
        "question": "Есть ли парковка у Технопарка Morion Digital?",
        "sources": [],
        "expect_abstain": True,
    }]
    item = check_all_questions("lexical", cases=cases)[0]
    assert item.expect_abstain is True
    assert item.found_sources == []
    assert item.abstained is True
    assert item.passed is True
    assert "Не нашёл" in (item.answer or "")

def test_summary_counts_failures():
    results = [
        QuestionResult(id="A", question="q", expected_sources=["x"], found_sources=["x"], hit_at_1=True, hit_at_3=True),
        QuestionResult(id="B", question="q", expected_sources=["y"], found_sources=["z"], hit_at_1=False, hit_at_3=False),
        QuestionResult(id="C", question="q", expected_sources=["y"], error="boom"),
        QuestionResult(id="N1", question="q", expected_sources=[], expect_abstain=True, abstained=True),
        QuestionResult(id="N2", question="q", expected_sources=[], expect_abstain=True, found_sources=["x"]),
    ]
    summary = summarize(results)
    assert summary["total"] == 5
    assert summary["answerable"] == 3
    assert summary["hit_at_1"] == 1
    assert summary["hit_at_3"] == 1
    assert summary["abstain_total"] == 2
    assert summary["abstain_correct"] == 1
    assert summary["failed_ids"] == ["B", "C", "N2"]
    assert summary["errors"] == 1

def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        check_all_questions("faiss")
