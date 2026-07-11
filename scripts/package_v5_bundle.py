"""Write and hash a deployable v5 two-adapter bundle manifest."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from apush_frq_grader_slm.training_v5 import write_bundle_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--inherited-base", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Write a temporary manifest pointing at artifacts instead of copying them.",
    )
    args = parser.parse_args()
    manifest_path = args.bundle / "v5_bundle.json"
    if manifest_path.exists() and not args.overwrite:
        print(manifest_path)
        return
    if args.bundle.exists() and any(args.bundle.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Bundle directory is not empty: {args.bundle}")
        shutil.rmtree(args.bundle)
    args.bundle.mkdir(parents=True, exist_ok=True)
    for source in (args.inherited_base, args.scorer, args.feedback):
        if not source.is_dir():
            raise FileNotFoundError(source)
    if args.reference_only:
        inherited, scorer, feedback = args.inherited_base, args.scorer, args.feedback
    else:
        inherited = args.bundle / "inherited_base"
        scorer = args.bundle / "scorer"
        feedback = args.bundle / "feedback"
        shutil.copytree(args.inherited_base, inherited)
        shutil.copytree(args.scorer, scorer)
        shutil.copytree(args.feedback, feedback)
    manifest = write_bundle_manifest(
        args.bundle,
        inherited_base=inherited,
        scorer_adapter=scorer,
        feedback_adapter=feedback,
    )
    print(args.bundle / "v5_bundle.json")
    print(manifest)


if __name__ == "__main__":
    main()
