"""Download and manifest Tom Richey APUSH LEQ sample PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TOM_RICHEY_SOURCES = [
    {
        "url": "https://www.tomrichey.net/uploads/3/2/1/0/32100773/2024_apush_leq_2_%5Bset_1%5D_-_causes_of_conflict_between_native_americans_and_europeans__1_.pdf",
        "filename": "tomrichey_2024_leq2_set1.pdf",
        "year": 2024,
        "leq_num": 2,
        "set": 1,
        "prompt": (
            "Evaluate the relative importance of causes of conflict between Europeans "
            "and Native Americans in the period from 1500 to 1763."
        ),
    },
    {
        "url": "https://www.tomrichey.net/uploads/3/2/1/0/32100773/2024_apush_leq_3_%5Bset_1%5D_-_causes_of_changes_in_national_culture__1800-1848_.pdf",
        "filename": "tomrichey_2024_leq3_set1.pdf",
        "year": 2024,
        "leq_num": 3,
        "set": 1,
        "prompt": (
            "Evaluate the extent to which economic development contributed to the growth "
            "of a distinct national culture in the United States in the period 1800-1848."
        ),
    },
    {
        "url": "https://www.tomrichey.net/uploads/3/2/1/0/32100773/2024_apush_leq_4_%5Bset_1%5D_-_effects_of_movements_for_social_change_%5B1945-1980%5D.pdf",
        "filename": "tomrichey_2024_leq4_set1.pdf",
        "year": 2024,
        "leq_num": 4,
        "set": 1,
        "prompt": (
            "Evaluate the extent to which movements for social change in the period "
            "from 1945 to 1980 contributed to greater equality in the United States."
        ),
    },
    {
        "url": "https://www.tomrichey.net/uploads/3/2/1/0/32100773/2022_apush_leq_2_sample_responses.pdf",
        "filename": "tomrichey_2022_leq2_responses.pdf",
        "year": 2022,
        "leq_num": 2,
        "set": 1,
        "prompt": (
            "Evaluate the extent to which the settlement of British North America "
            "was primarily driven by economic motivations in the period 1607-1754."
        ),
    },
    {
        "url": "https://www.tomrichey.net/uploads/3/2/1/0/32100773/sample_leq_responses_-_road_to_american_revolution_-_apush.pdf",
        "filename": "tomrichey_revolution_road_leq.pdf",
        "year": 2020,
        "leq_num": 2,
        "set": 1,
        "prompt": (
            "Evaluate the extent to which the American Revolution was caused by disputes "
            "over economic policy in the period from 1754 to 1776."
        ),
    },
]


def download_file(url: str, dest: Path, *, timeout: int = 60) -> tuple[int, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "apush-frq-grader-slm/0.1"})
    with urlopen(request, timeout=timeout) as response:
        data = response.read()
    dest.write_bytes(data)
    return len(data), hashlib.sha256(data).hexdigest()


def build_manifest(output_dir: Path) -> list[dict]:
    manifest: list[dict] = []
    for entry in TOM_RICHEY_SOURCES:
        dest = output_dir / entry["filename"]
        record = {**entry, "source": "tom_richey", "local_path": str(dest)}
        try:
            size, sha256 = download_file(entry["url"], dest)
            record.update({"status": "downloaded", "bytes": size, "sha256": sha256})
        except (HTTPError, URLError, TimeoutError) as exc:
            record.update({"status": "failed", "error": str(exc)})
        manifest.append(record)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.output_dir)
    path = args.output_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok = sum(1 for row in manifest if row.get("status") == "downloaded")
    print(f"Wrote {path} ({ok}/{len(manifest)} downloaded)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Catalog Tom Richey LEQ PDFs.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/raw/tomrichey"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
