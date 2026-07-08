"""Catalog curated Quizlet APUSH LEQ sets and optionally fetch term JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Curated public sets with LEQ essays, outlines, or multi-card responses.
QUIZLET_SETS = [
    {
        "set_id": "485501886",
        "title": "APUSH LEQ Spanish-American War turning point",
        "prompt": (
            "Evaluate the extent to which the Spanish-American War marked a turning point "
            "in American foreign policy."
        ),
    },
    {
        "set_id": "756614068",
        "title": "APUSH LEQ's thesis and CC practice",
    },
    {
        "set_id": "898358597",
        "title": "APUSH Periods 8/9 LEQ and DBQ Prompts",
    },
    {
        "set_id": "544429363",
        "title": "APUSH DBQ and LEQ",
    },
    {
        "set_id": "278103637",
        "title": "APUSH Period 7 LEQ",
    },
    {
        "set_id": "910875787",
        "title": "APUSH Final LEQ Prompts",
    },
    {
        "set_id": "196239678",
        "title": "APUSH possible LEQ's",
    },
    {
        "set_id": "452783417",
        "title": "APUSH Unit 3 LEQ Prompts",
    },
]


def fetch_set_terms(set_id: str, *, client_id: str, access_token: str | None = None) -> dict:
    params = {"per_page": 500, "client_id": client_id}
    headers = {"User-Agent": "apush-frq-grader-slm/0.1"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    url = f"https://api.quizlet.com/2.0/sets/{set_id}/terms?{urlencode(params)}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    terms = [
        {"term": row.get("term", ""), "definition": row.get("definition", "")}
        for row in payload.get("terms", [])
    ]
    return {"set_id": set_id, "terms": terms}


def build_manifest(output_dir: Path, *, fetch: bool = False) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    client_id = os.environ.get("QUIZLET_CLIENT_ID", "")
    access_token = os.environ.get("QUIZLET_ACCESS_TOKEN")
    manifest: list[dict] = []

    for entry in QUIZLET_SETS:
        set_id = entry["set_id"]
        record = {**entry, "source": "quizlet", "url": f"https://quizlet.com/{set_id}"}
        json_path = output_dir / f"{set_id}.json"
        record["local_path"] = str(json_path)

        if json_path.exists():
            record["status"] = "cached"
        elif fetch and client_id:
            try:
                payload = fetch_set_terms(set_id, client_id=client_id, access_token=access_token)
                payload.update({k: v for k, v in entry.items() if k != "set_id"})
                json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                record["status"] = "downloaded"
                record["term_count"] = len(payload.get("terms", []))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
        else:
            record["status"] = "manifest_only"
            if not client_id:
                record["note"] = "Set QUIZLET_CLIENT_ID to fetch; or copy JSON to local_path"
        manifest.append(record)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.output_dir, fetch=args.fetch)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path} ({len(manifest)} sets)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Catalog Quizlet APUSH LEQ sets.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/raw/quizlet"))
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch terms via Quizlet API (requires QUIZLET_CLIENT_ID)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
