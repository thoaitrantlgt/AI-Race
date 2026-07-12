from __future__ import annotations

import argparse
import json
import urllib.parse

import requests


QUERIES = [
    '"Tăng huyết áp" "I10" "ICD"',
    '"Bệnh tả" "A00" "ICD"',
    '"Danh mục ICD-10" "A00"',
    '"viêm phổi" "J18" "ICD10"',
    '"ICD-10" "Bộ Y tế" "A00"',
]


def github_code_search() -> None:
    for query in QUERIES:
        url = "https://api.github.com/search/code?q=" + urllib.parse.quote(query)
        response = requests.get(url, timeout=60, headers={"Accept": "application/vnd.github+json"})
        print("QUERY", query.encode("unicode_escape").decode("ascii"))
        print("STATUS", response.status_code)
        try:
            data = response.json()
        except Exception:
            print(response.text[:500])
            continue
        for item in data.get("items", [])[:10]:
            print(json.dumps({"name": item.get("name"), "path": item.get("path"), "html_url": item.get("html_url")}, ensure_ascii=True))
        if response.status_code != 200:
            print(json.dumps(data, ensure_ascii=True)[:1000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    github_code_search()


if __name__ == "__main__":
    main()
