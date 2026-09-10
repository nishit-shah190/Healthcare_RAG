import re

def normalize_text(text:str) -> str:
    return text.lower().strip()

def fact_coverage(
    answer:str,
    expected_facts: list[str],
) -> float:
    if not expected_facts:
        return 1.0
    normalized_answer = normalize_text(answer)
    matched_facts = 0

    for fact in expected_facts:
        if normalize_text(fact) in normalized_answer:
            matched_facts += 1
    return matched_facts/len(expected_facts)

ABSTENTION_PHASE = ("i do not have enough information")

def detected_abstention(answer: str) -> bool:
    normalized_answer = normalize_text(answer)

    return ABSTENTION_PHASE in normalized_answer

def abstention_correct(
    answer:str,
    should_abstain:bool,
) -> float:
    actual_abstention = detected_abstention(answer)

    return float(
        actual_abstention == should_abstain
    )

def has_citation(answer: str) -> bool:
    citation_pattern = r"\[\d+\]"

    return bool(re.search(citation_pattern, answer))

def citation_presence_score(
    answer:str,
    should_abstain: bool,
) -> float:
    if should_abstain:
        return 1.0
    return float(has_citation(answer))
