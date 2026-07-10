"""Download and manifest College Board APUSH LEQ APC PDFs (2015–2025).

2021–2025 produced the frozen 72-essay eval set. Older years (2015–2020) are
enumerated to source *net-new* seed essays for realistic generation; their URL
slugs and in-PDF header layouts vary, so some downloads/parses will fail (those
are recorded as status="failed" in the manifest, not fatal)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from apush_frq_grader_slm.golden import load_permission_record, require_permission

BASE_URL = "https://apcentral.collegeboard.org/media/pdf"
YEARS = (15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
LEQ_NUMS = (2, 3, 4)
SETS = (1, 2)

# Optional Tom Richey mirrors for parse validation (same CB content, easier labels).
TOM_RICHEY_FIXTURES = [
    {
        "url": "https://www.tomrichey.net/uploads/3/2/1/0/32100773/2024_apush_leq_3_[set_1]_-_causes_of_changes_in_national_culture__1800-1848_.pdf",
        "filename": "tomrichey_2024_leq3_set1.pdf",
        "source": "tom_richey",
    },
]


def enumerate_sources() -> list[dict]:
    sources: list[dict] = []
    for year_suffix in YEARS:
        for leq_num in LEQ_NUMS:
            for set_num in SETS:
                filename = f"ap{year_suffix:02d}-apc-us-history-leq{leq_num}-set-{set_num}.pdf"
                sources.append(
                    {
                        "year": 2000 + year_suffix,
                        "leq_num": leq_num,
                        "set": set_num,
                        "filename": filename,
                        "url": f"{BASE_URL}/{filename}",
                        "source": "ap_central",
                    }
                )
    for fixture in TOM_RICHEY_FIXTURES:
        sources.append(fixture)
    return sources


def download_file(url: str, dest: Path, *, timeout: int = 60) -> tuple[int, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "apush-frq-grader-slm/0.1"})
    with urlopen(request, timeout=timeout) as response:
        data = response.read()
    dest.write_bytes(data)
    sha256 = hashlib.sha256(data).hexdigest()
    return len(data), sha256


def build_manifest(output_dir: Path, *, include_tomrichey: bool = False) -> list[dict]:
    manifest: list[dict] = []
    sources = enumerate_sources()
    if not include_tomrichey:
        sources = [entry for entry in sources if entry.get("source") == "ap_central"]

    for entry in sources:
        filename = entry["filename"]
        dest = output_dir / filename
        record = {**entry, "local_path": str(dest)}
        try:
            size, sha256 = download_file(entry["url"], dest)
            record.update(
                {
                    "status": "downloaded",
                    "bytes": size,
                    "sha256": sha256,
                }
            )
        except (HTTPError, URLError, TimeoutError) as exc:
            record.update({"status": "failed", "error": str(exc)})
        manifest.append(record)
    return manifest


def main() -> None:
    args = parse_args()
    require_permission(load_permission_record(args.permission_record), "storage")
    manifest = build_manifest(args.output_dir, include_tomrichey=args.include_tomrichey)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    downloaded = sum(1 for row in manifest if row.get("status") == "downloaded")
    failed = sum(1 for row in manifest if row.get("status") == "failed")
    print(f"Manifest written to {manifest_path}")
    print(f"Downloaded: {downloaded}, Failed: {failed}, Total: {len(manifest)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Catalog and download AP Central LEQ PDFs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/raw/ap_central"),
        help="Directory for PDFs and manifest.json",
    )
    parser.add_argument(
        "--permission-record",
        type=Path,
        default=Path("config/college_board_permission.json"),
    )
    parser.add_argument(
        "--include-tomrichey",
        action="store_true",
        help="Also download Tom Richey parse-test fixtures",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
