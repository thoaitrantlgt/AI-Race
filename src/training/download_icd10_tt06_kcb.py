from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://ccs.whiteneuron.com/api/ICD10_TT06"
CODE_TRAILING_MARKS = re.compile(r"[†*]+$")


def _get_json(session: requests.Session, url: str, params: dict[str, str], retries: int = 4) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} with params={params}: {last_error}")


def _node_row(node: dict[str, Any]) -> tuple[str, str, str, str, str] | None:
    if node.get("model") == "chapter":
        return None
    data = node.get("data") or {}
    code = str(data.get("code") or data.get("id") or node.get("id") or "").strip()
    code = CODE_TRAILING_MARKS.sub("", code).strip()
    name = str(data.get("name") or "").strip()
    if not code or not name:
        return None
    return (name, name, f"ICD10:{code}", "ICD10_TT06", "diagnosis")


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "AI_RACE terminology downloader",
            "Referer": "https://icd.kcb.vn/icd-10-tt06/icd10-tt06",
        }
    )
    return session


def _fetch_children(node: dict[str, Any]) -> list[dict[str, Any]]:
    if node.get("is_leaf"):
        return []
    model = str(node.get("model") or "").strip()
    node_id = str(node.get("id") or (node.get("data") or {}).get("id") or "").strip()
    if not model or not node_id:
        return []
    session = _make_session()
    payload = _get_json(session, f"{BASE_URL}/childs/{model}", {"id": node_id, "lang": "vi"})
    return payload.get("data") or []


def download_icd10_tt06(output_path: str | Path, sleep_s: float = 0.0, workers: int = 12) -> None:
    session = _make_session()

    root = _get_json(session, f"{BASE_URL}/root", {"lang": "vi"})
    frontier: list[dict[str, Any]] = root.get("data") or []
    seen_nodes: set[tuple[str, str]] = set()
    rows: dict[tuple[str, str], tuple[str, str, str, str, str]] = {}
    scanned = 0
    level = 0

    while frontier:
        level += 1
        current: list[dict[str, Any]] = []
        for node in frontier:
            model = str(node.get("model") or "").strip()
            node_id = str(node.get("id") or (node.get("data") or {}).get("id") or "").strip()
            if not model or not node_id or (model, node_id) in seen_nodes:
                continue
            seen_nodes.add((model, node_id))
            current.append(node)

            row = _node_row(node)
            if row:
                rows[(row[0].casefold(), row[2])] = row

        scanned += len(current)
        print(f"Level {level}: {len(current)} nodes, scanned={scanned}, rows={len(rows)}", flush=True)
        frontier = []
        non_leaf = [node for node in current if not node.get("is_leaf")]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(_fetch_children, node) for node in non_leaf]
            for future in as_completed(futures):
                frontier.extend(future.result())
                if sleep_s:
                    time.sleep(sleep_s)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# source=https://icd.kcb.vn/icd-10-tt06/icd10-tt06\n")
        handle.write("# api_base=https://ccs.whiteneuron.com/api/ICD10_TT06\n")
        handle.write("# regulation=06/2026/TT-BYT\n")
        for row in sorted(rows.values(), key=lambda item: (item[2], item[0].casefold())):
            handle.write("\t".join(row) + "\n")
    tmp.replace(out)
    print(f"Scanned {scanned} ICD-10 TT06 nodes", flush=True)
    print(f"Wrote {len(rows)} ICD-10 TT06 aliases to {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/terminology/custom_aliases/icd10_tt06_2026_kcb.tsv")
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    download_icd10_tt06(args.output, args.sleep_s, args.workers)


if __name__ == "__main__":
    main()
