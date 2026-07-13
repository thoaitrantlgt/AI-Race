from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import build_terminology_store, load_config
from src.preprocess.section_parser import get_section_for_offset, parse_sections
from src.preprocess.text_normalizer import normalize_for_matching


ALPHA_TOKEN_RE = re.compile(r"^[A-Za-zÀ-ỹĐđ]{5,}$")
HISTORICAL_SECTIONS = {"past_history", "medication_history"}
HISTORICAL_CONTEXT_CUES = (
    "tiền sử",
    "trước đây",
    "trước khi nhập viện",
    "lần nhập viện trước",
    "thuốc đang dùng trước",
    "đang dùng tại nhà",
    "ở nhà bệnh nhân đã",
    "đã sử dụng",
    "đã điều trị",
)
HISTORICAL_HEADING_CUES = ("các sự kiện trước khi nhập viện",)


def infer_repaired_assertions(raw_text: str, sections, start: int, end: int, fallback: list[str]) -> list[str]:
    section = get_section_for_offset(sections, start, end)
    context = normalize_for_matching(raw_text[max(0, start - 240) : min(len(raw_text), end + 100)])
    preceding_section_context = normalize_for_matching(raw_text[max(0, start - 900) : start])
    is_historical = section in HISTORICAL_SECTIONS or any(
        normalize_for_matching(cue) in context for cue in HISTORICAL_CONTEXT_CUES
    ) or any(
        normalize_for_matching(cue) in preceding_section_context for cue in HISTORICAL_HEADING_CUES
    )
    if is_historical:
        return ["isHistorical"]
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-dir", default="data/input")
    parser.add_argument("--config", default="configs/llm_pseudo_hybrid.yaml")
    args = parser.parse_args()
    store = build_terminology_store(load_config(args.config))

    with zipfile.ZipFile(args.input) as source_zip:
        names = [name for name in source_zip.namelist() if name.endswith(".json")]
        records = {name: json.loads(source_zip.read(name).decode("utf-8")) for name in names}

    names_by_candidate: dict[str, set[str]] = defaultdict(set)
    for entities in records.values():
        for entity in entities:
            text = entity.get("text", "").strip().lower()
            if entity.get("type") != "THUỐC" or not ALPHA_TOKEN_RE.fullmatch(text):
                continue
            for candidate in entity.get("candidates", []):
                names_by_candidate[candidate].add(text)

    canonical_to_candidates: dict[str, list[str]] = defaultdict(list)
    for candidate, names_for_id in names_by_candidate.items():
        canonical = min(names_for_id, key=lambda value: (len(value), value))
        canonical_to_candidates[canonical].append(candidate)
    canonical_names = sorted(canonical_to_candidates, key=len, reverse=True)

    repaired_count = 0
    created_count = 0
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as output_zip:
        for name, entities in records.items():
            input_path = Path(args.input_dir) / f"{Path(name).stem}.txt"
            raw_text = input_path.read_text(encoding="utf-8") if input_path.exists() else ""
            sections = parse_sections(raw_text)
            repaired: list[dict] = []
            for entity in entities:
                if entity.get("type") != "THUỐC":
                    repaired.append(entity)
                    continue
                text = entity.get("text", "")
                first_token = text.split(maxsplit=1)[0].lower() if text else ""
                matches: list[tuple[int, int, str]] = []
                for canonical in canonical_names:
                    search_from = 0
                    while True:
                        start = first_token.find(canonical, search_from)
                        if start < 0:
                            break
                        end = start + len(canonical)
                        if not any(max(start, old_start) < min(end, old_end) for old_start, old_end, _ in matches):
                            matches.append((start, end, canonical))
                        search_from = start + 1
                matches.sort()
                boundaries = [(0, len(first_token))]
                for start, end, _ in matches:
                    next_boundaries: list[tuple[int, int]] = []
                    for gap_start, gap_end in boundaries:
                        if end <= gap_start or gap_end <= start:
                            next_boundaries.append((gap_start, gap_end))
                            continue
                        if gap_start < start:
                            next_boundaries.append((gap_start, start))
                        if end < gap_end:
                            next_boundaries.append((end, gap_end))
                    boundaries = next_boundaries
                for gap_start, gap_end in boundaries:
                    token = first_token[gap_start:gap_end]
                    if len(token) < 5:
                        continue
                    entries = [entry for entry in store.lookup_exact(token) if entry.source == "RxNorm" or entry.semantic_type == "drug"]
                    if not entries:
                        continue
                    candidate_ids = []
                    for entry in entries:
                        candidate_id = entry.concept_id[2:] if entry.concept_id.startswith("Rx") else entry.concept_id
                        if candidate_id not in candidate_ids:
                            candidate_ids.append(candidate_id)
                    canonical_to_candidates[token] = candidate_ids[:2]
                    matches.append((gap_start, gap_end, token))
                matches.sort()
                token_is_concatenated = bool(matches) and (
                    len(matches) >= 2 or any(start == 0 and end < len(first_token) for start, end, _ in matches)
                )
                if not token_is_concatenated:
                    repaired.append(entity)
                    continue
                base_start = int(entity["position"][0])
                assertions = infer_repaired_assertions(
                    raw_text,
                    sections,
                    base_start,
                    int(entity["position"][1]),
                    list(entity.get("assertions", [])),
                )
                for start, end, canonical in matches:
                    replacement = {
                        "text": text[start:end],
                        "position": [base_start + start, base_start + end],
                        "type": "THUỐC",
                        "assertions": assertions,
                        "candidates": list(canonical_to_candidates[canonical])[:2],
                    }
                    repaired.append(replacement)
                    created_count += 1
                repaired_count += 1

            deduplicated: list[dict] = []
            seen: set[tuple] = set()
            for entity in repaired:
                key = tuple(entity["position"]), entity["type"]
                if key in seen:
                    continue
                seen.add(key)
                deduplicated.append(entity)
            output_zip.writestr(name, json.dumps(deduplicated, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"Repaired {repaired_count} concatenated drug spans into {created_count} canonical spans")


if __name__ == "__main__":
    main()
