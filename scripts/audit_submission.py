from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.diagnosis_symptom_extractor import SYMPTOMS
from src.extraction.lab_extractor import LAB_NAMES


WORD_RE = re.compile(r"[A-Za-zÀ-ỹĐđ0-9]")
TRAILING_FRAGMENT_RE = re.compile(
    r"\b(?:trong|cho|và|với|của|tại|do|để|khi|ngày|lần|khả|ng|kh|v)\s*$",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("--input-dir", default="data/input")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    findings: list[tuple[str, str, str, tuple[int, int], str]] = []
    type_counts: Counter[str] = Counter()
    empty_candidates: Counter[str] = Counter()
    assertion_counts: Counter[str] = Counter()
    uncovered: list[tuple[str, str, tuple[int, int], str]] = []

    with zipfile.ZipFile(args.submission) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".json"):
                continue
            raw_path = Path(args.input_dir) / f"{Path(name).stem}.txt"
            raw = raw_path.read_text(encoding="utf-8")
            entities = json.loads(archive.read(name).decode("utf-8"))
            spans_by_type: dict[str, list[tuple[int, int]]] = {}
            for entity in entities:
                spans_by_type.setdefault(entity["type"], []).append(tuple(entity["position"]))

            for typ, terms in (("TRIỆU_CHỨNG", SYMPTOMS), ("TÊN_XÉT_NGHIỆM", LAB_NAMES)):
                pattern = re.compile(
                    rf"(?<!\w)({'|'.join(re.escape(term) for term in sorted(set(terms), key=len, reverse=True))})(?!\w)",
                    re.IGNORECASE,
                )
                for match in pattern.finditer(raw):
                    if not any(left <= match.start() and match.end() <= right for left, right in spans_by_type.get(typ, [])):
                        uncovered.append((name, typ, (match.start(), match.end()), match.group(0)))

            for entity in entities:
                text = entity["text"]
                start, end = entity["position"]
                typ = entity["type"]
                type_counts[typ] += 1
                if typ in {"CHẨN_ĐOÁN", "THUỐC"} and not entity.get("candidates"):
                    empty_candidates[typ] += 1
                assertion_counts.update(entity.get("assertions", []))
                if start and WORD_RE.match(raw[start - 1]) and text and WORD_RE.match(text[0]):
                    findings.append((name, "midword-left", typ, (start, end), raw[max(0, start - 20) : min(len(raw), end + 30)]))
                if end < len(raw) and text and WORD_RE.match(text[-1]) and WORD_RE.match(raw[end]):
                    findings.append((name, "midword-right", typ, (start, end), raw[max(0, start - 20) : min(len(raw), end + 30)]))
                if TRAILING_FRAGMENT_RE.search(text):
                    findings.append((name, "trailing-fragment", typ, (start, end), raw[max(0, start - 20) : min(len(raw), end + 30)]))
                if "\n" in text:
                    findings.append((name, "contains-newline", typ, (start, end), repr(text)))

            ordered = sorted(entities, key=lambda item: tuple(item["position"]))
            for left, right in zip(ordered, ordered[1:]):
                left_start, left_end = left["position"]
                right_start, right_end = right["position"]
                if right_start < left_end:
                    findings.append(
                        (
                            name,
                            "overlap",
                            f"{left['type']}+{right['type']}",
                            (right_start, min(left_end, right_end)),
                            raw[max(0, left_start - 15) : min(len(raw), right_end + 15)],
                        )
                    )

    print("type_counts", dict(type_counts))
    print("empty_candidates", dict(empty_candidates))
    print("assertions", dict(assertion_counts))
    print("finding_counts", dict(Counter(item[1] for item in findings)))
    print("uncovered_counts", dict(Counter(item[1] for item in uncovered)))
    for item in uncovered[: args.limit]:
        print("uncovered", item)
    for finding in findings[: args.limit]:
        print(finding)


if __name__ == "__main__":
    main()
