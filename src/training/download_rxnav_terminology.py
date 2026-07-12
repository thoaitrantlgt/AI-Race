from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


DEFAULT_TTYS = ["IN", "PIN", "BN", "SCD", "SBD", "GPCK", "BPCK", "MIN", "SCDG", "SBDG", "SCDC"]


def download_rxnav_terminology(output_path: str | Path, ttys: list[str] | None = None, sleep_s: float = 0.2) -> None:
    ttys = ttys or DEFAULT_TTYS
    rows: dict[tuple[str, str], tuple[str, str, str, str, str]] = {}
    version = requests.get("https://rxnav.nlm.nih.gov/REST/version.json", timeout=60).json()
    for tty in ttys:
        url = f"https://rxnav.nlm.nih.gov/REST/allconcepts.json?tty={tty}"
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        concepts = response.json().get("minConceptGroup", {}).get("minConcept", [])
        for concept in concepts:
            rxcui = str(concept.get("rxcui", "")).strip()
            name = str(concept.get("name", "")).strip()
            concept_tty = str(concept.get("tty", tty)).strip()
            if not rxcui or not name:
                continue
            rows[(name.lower(), rxcui)] = (name, name, f"Rx{rxcui}", "RxNorm", "drug")
        print(f"{tty}: {len(concepts)} concepts")
        time.sleep(sleep_s)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# source=RxNav\n")
        handle.write(f"# rxnorm_version={version.get('version')}\n")
        handle.write(f"# api_version={version.get('apiVersion')}\n")
        for row in sorted(rows.values(), key=lambda item: item[0].lower()):
            handle.write("\t".join(row) + "\n")
    print(f"Wrote {len(rows)} RxNorm aliases to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/terminology/custom_aliases/rxnorm_rxnav_2026.tsv")
    parser.add_argument("--ttys", nargs="*", default=DEFAULT_TTYS)
    args = parser.parse_args()
    download_rxnav_terminology(args.output, args.ttys)


if __name__ == "__main__":
    main()
