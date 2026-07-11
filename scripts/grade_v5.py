"""Grade one APUSH LEQ with a packaged v5 two-pass bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.inference_v5 import V5BundleGrader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--essay", required=True)
    parser.add_argument("--verify-hashes", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    grader = V5BundleGrader(args.bundle, verify_hashes=args.verify_hashes)
    print(json.dumps(grader.grade(args.prompt, args.essay), ensure_ascii=False))


if __name__ == "__main__":
    main()
