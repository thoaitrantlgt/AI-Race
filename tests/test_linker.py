from src.linking.alias_loader import load_custom_aliases
from src.linking.candidate_generator import generate_candidates
from src.linking.terminology_store import TerminologyStore


def test_builtin_alias_linker():
    store = TerminologyStore()
    load_custom_aliases("missing", store)
    candidates = generate_candidates("tylenol", "drug", "", store)
    assert any(candidate.concept_id == "Rx161" for candidate in candidates)
