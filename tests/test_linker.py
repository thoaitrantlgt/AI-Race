from src.linking.alias_loader import load_custom_aliases
from src.linking.candidate_generator import generate_candidates
from src.linking.rxnorm_loader import load_rxnorm_to_store
from src.linking.terminology_store import TerminologyStore


def test_builtin_alias_linker():
    store = TerminologyStore()
    load_custom_aliases("missing", store)
    candidates = generate_candidates("tylenol", "drug", "", store)
    assert any(candidate.concept_id == "Rx161" for candidate in candidates)


def test_rxnorm_loader_keeps_strength_components_and_deduplicates(tmp_path):
    root = tmp_path / "rxnorm_2026"
    root.mkdir()

    def row(rxcui: str, tty: str, name: str) -> str:
        fields = [rxcui, "ENG", "", "", "", "", "", "", "", "", "", "RXNORM", tty, rxcui, name, "", "N", ""]
        return "|".join(fields) + "|\n"

    (root / "RXNCONSO.RRF").write_text(
        row("315643", "SCDC", "chlorpheniramine 0.4 MG/ML")
        + row("315643", "SCDC", "chlorpheniramine 0.4 MG/ML")
        + row("315643", "SY", "chlorpheniramine strength synonym"),
        encoding="utf-8",
    )
    store = TerminologyStore()
    load_rxnorm_to_store(root, store)

    matches = store.lookup_exact("chlorpheniramine 0.4 mg/ml")
    assert [(entry.concept_id, entry.version) for entry in matches] == [("Rx315643", "2026-07-06")]


def test_custom_alias_loader_can_skip_redundant_rxnav_bulk(tmp_path):
    (tmp_path / "rxnorm_rxnav_2026.tsv").write_text(
        "bulk drug\tbulk drug\tRx100\tRxNorm\tdrug\n", encoding="utf-8"
    )
    (tmp_path / "rxnorm_seed.tsv").write_text(
        "seed drug\tseed drug\tRx200\tRxNorm\tdrug\n", encoding="utf-8"
    )
    store = TerminologyStore()
    load_custom_aliases(tmp_path, store, skip_rxnorm_bulk=True)

    assert not store.lookup_exact("bulk drug")
    assert store.lookup_exact("seed drug")[0].concept_id == "Rx200"
