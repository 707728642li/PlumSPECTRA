from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


DOIS = [
    "10.1016/j.postharvbio.2007.06.024",
    "10.1016/j.postharvbio.2020.111139",
    "10.1016/j.postharvbio.2020.111202",
    "10.1016/j.chemolab.2021.104287",
    "10.1016/j.chemolab.2023.104924",
    "10.1016/j.saa.2024.124003",
    "10.1016/j.postharvbio.2024.112783",
    "10.1016/j.saa.2025.126122",
    "10.1016/j.compag.2026.112186",
    "10.1016/j.foodchem.2026.148344",
    "10.1016/j.foodchem.2025.145387",
    "10.1093/fqsafe/fyac068",
    "10.5935/0103-5053.20130172",
    "10.1111/jfpe.13597",
    "10.11924/j.issn.1000-6850.casb2024-0452",
    "10.1016/j.saa.2026.128279",
    "10.1016/j.scienta.2014.01.002",
    "10.1016/j.scienta.2026.114793",
    "10.3389/fpls.2023.1128993",
    "10.1016/j.foodcont.2024.110823",
    "10.1016/j.aiia.2025.12.003",
    "10.1038/s41438-021-00560-9",
    "10.1016/j.saa.2023.123151",
    "10.1111/jfpp.16504",
    "10.1016/j.meafoo.2025.100246",
    "10.1021/ac60214a047",
    "10.1016/j.jcm.2016.02.012",
    "10.2307/2532051",
    "10.1561/2200000101",
    "10.1109/CVPR.2016.90",
    "10.48550/arXiv.1706.03762",
]


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", value or "")).strip()


def fetch(doi: str) -> dict[str, object]:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(url, headers={"User-Agent": "NIRsPlumResearch/1.0 (mailto:research@example.invalid)"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["message"]


def year_from(record: dict[str, object]) -> int | None:
    for key in ["published-print", "published-online", "published", "issued"]:
        if key in record:
            parts = record[key].get("date-parts", [])
            if parts and parts[0]:
                return int(parts[0][0])
    return None


def cite_key(record: dict[str, object], year: int | None, index: int) -> str:
    authors = record.get("author") or []
    family = clean(authors[0].get("family", "ref")) if authors else "ref"
    family = re.sub(r"[^A-Za-z0-9]", "", family) or "ref"
    return f"{family.lower()}{year or 'nd'}_{index:02d}"


def bib_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    bib_entries: list[str] = []
    failures: list[dict[str, str]] = []
    for index, doi in enumerate(DOIS, start=1):
        try:
            record = fetch(doi)
        except Exception as exc:
            failures.append({"doi": doi, "error": str(exc)})
            continue
        raw[doi] = record
        year = year_from(record)
        authors = record.get("author") or []
        author_text = "; ".join(
            clean(" ".join(filter(None, [author.get("given", ""), author.get("family", "")])))
            for author in authors
        )
        title = clean((record.get("title") or [""])[0])
        journal = clean((record.get("container-title") or [""])[0])
        volume = clean(str(record.get("volume", "")))
        issue = clean(str(record.get("issue", "")))
        pages = clean(str(record.get("page", record.get("article-number", ""))))
        key = cite_key(record, year, index)
        rows.append(
            {
                "key": key,
                "doi": doi,
                "year": year,
                "authors": author_text,
                "title": title,
                "journal": journal,
                "volume": volume,
                "issue": issue,
                "pages_or_article": pages,
                "url": f"https://doi.org/{doi}",
            }
        )
        bib_authors = " and ".join(
            clean(" ".join(filter(None, [author.get("family", ""), author.get("given", "")])))
            for author in authors
        )
        fields = {
            "author": bib_authors,
            "title": "{" + title + "}",
            "journal": journal,
            "year": str(year or ""),
            "volume": volume,
            "number": issue,
            "pages": pages,
            "doi": doi,
            "url": f"https://doi.org/{doi}",
        }
        body = ",\n".join(f"  {name} = {{{bib_escape(value)}}}" for name, value in fields.items() if value)
        bib_entries.append(f"@article{{{key},\n{body}\n}}")
        time.sleep(0.08)

    pd.DataFrame(rows).to_csv(output / "reference_metadata.csv", index=False)
    (output / "references.bib").write_text("\n\n".join(bib_entries) + "\n", encoding="utf-8")
    (output / "crossref_records.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (output / "fetch_report.json").write_text(
        json.dumps({"requested": len(DOIS), "retrieved": len(rows), "failures": failures}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"requested": len(DOIS), "retrieved": len(rows), "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
