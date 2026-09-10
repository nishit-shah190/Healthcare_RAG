from app.evaluation.generation_metrics import(
    fact_coverage,
    detected_abstention,
    abstention_correct,
    has_citation,
)


def test_fact_coverage():
    answer = ("Treatment should consider kidney function "
        "and cardiovascular disease.")
    expected_facts = [
        "kidney function",
        "cardiovascular disease",
        "hypoglycemia risk",
    ]
    assert fact_coverage(
        answer, expected_facts
    ) == 2/3

def test_detected_abstention():
    answer = (
        "I do not have enough information "
        "in the retrieved knowledge base "
        "to answer this reliably."
    )

    assert detected_abstention(answer) is True

def test_abstention_correct():
    answer = (
        "I do not have enough information "
        "to answer this reliably."
    )

    assert abstention_correct(
        answer,
        should_abstain=True,
    ) == 1.0
def test_has_citation():

    answer = (
        "Kidney function should be considered [1]."
    )

    assert has_citation(answer) is True


    