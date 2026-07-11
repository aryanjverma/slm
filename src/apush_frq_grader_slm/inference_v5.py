"""Load and run a v5 inherited-base, two-adapter grading bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from apush_frq_grader_slm.baselines import ResponseAdapter
from apush_frq_grader_slm.prompts_v5 import (
    V5_FEEDBACK_SYSTEM_PROMPT,
    V5_SCORER_SYSTEM_PROMPT,
    format_v5_feedback_user_message,
    format_v5_scorer_user_message,
)
from apush_frq_grader_slm.rubric import CRITERIA, SCORE_RANGES, compute_total
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.structured_output_v3 import extract_balanced_json_objects
from apush_frq_grader_slm.training_v5 import resolve_manifest_path, sha256_tree


def _load_bundle_prompt(
    root: Path,
    manifest: Mapping[str, Any],
    key: str,
    default: str,
    *,
    verify_hashes: bool,
) -> str:
    prompts = manifest.get("prompts")
    if not isinstance(prompts, Mapping) or key not in prompts:
        return default
    entry = prompts[key]
    if not isinstance(entry, Mapping) or "path" not in entry:
        return default
    path = resolve_manifest_path(root, entry)  # type: ignore[arg-type]
    if not path.is_file():
        return default
    if verify_hashes and entry.get("sha256") and sha256_tree(path) != entry["sha256"]:
        raise ValueError(f"V5 bundle hash mismatch for prompt {key}: {path}")
    return path.read_text(encoding="utf-8").rstrip("\n")


class V5BundleGrader:
    """One merged base with independently selectable scorer and feedback adapters."""

    def __init__(self, bundle_root: Path | str, *, verify_hashes: bool = True) -> None:
        root = Path(bundle_root)
        manifest = json.loads((root / "v5_bundle.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "apush-frq-grader-v5":
            raise ValueError("Not an APUSH v5 bundle manifest")
        paths = {
            key: resolve_manifest_path(root, manifest[key])
            for key in ("inherited_base", "scorer", "feedback")
        }
        if verify_hashes:
            for key, path in paths.items():
                observed = sha256_tree(path)
                if observed != manifest[key]["sha256"]:
                    raise ValueError(f"V5 bundle hash mismatch for {key}: {path}")
        self.scorer_system_prompt = _load_bundle_prompt(
            root, manifest, "scorer_system", V5_SCORER_SYSTEM_PROMPT, verify_hashes=verify_hashes
        )
        self.feedback_system_prompt = _load_bundle_prompt(
            root,
            manifest,
            "feedback_system",
            V5_FEEDBACK_SYSTEM_PROMPT,
            verify_hashes=verify_hashes,
        )
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError('Inference dependencies missing; install ".[train]"') from exc

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(paths["inherited_base"])
        base = AutoModelForCausalLM.from_pretrained(
            paths["inherited_base"], torch_dtype=dtype, low_cpu_mem_usage=True
        )
        self.model = PeftModel.from_pretrained(base, paths["scorer"], adapter_name="scorer")
        self.model.load_adapter(paths["feedback"], adapter_name="feedback")
        self.model = self.model.to(self.device)
        self.model.eval()

    def _generate(self, adapter: str, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        self.model.set_adapter(adapter)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()

    def grade(self, prompt: str, essay: str) -> dict[str, Any]:
        return grade_two_pass(
            prompt,
            essay,
            self._generate,
            scorer_system=self.scorer_system_prompt,
            feedback_system=self.feedback_system_prompt,
        )


class V5BundleAdapter(ResponseAdapter):
    """ResponseAdapter wrapper around ``V5BundleGrader`` for litmus/eval entrypoints."""

    def __init__(
        self,
        bundle_root: Path | str,
        *,
        verify_hashes: bool = True,
        name: str = "apush_frq_grader_v5",
    ) -> None:
        super().__init__(name=name)
        self.bundle_root = Path(bundle_root)
        self.verify_hashes = verify_hashes
        self._grader: V5BundleGrader | None = None

    def _get_grader(self) -> V5BundleGrader:
        if self._grader is None:
            self._grader = V5BundleGrader(self.bundle_root, verify_hashes=self.verify_hashes)
        return self._grader

    def respond(self, case: FRQCase) -> str:
        prediction = self._get_grader().grade(case.prompt, case.student_response)
        return json.dumps(prediction, ensure_ascii=True, separators=(",", ":"))


def grade_two_pass(
    prompt: str,
    essay: str,
    generate: Callable[[str, list[dict[str, str]], int], str],
    *,
    scorer_system: str = V5_SCORER_SYSTEM_PROMPT,
    feedback_system: str = V5_FEEDBACK_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Preserve the external scores/total/feedback contract across two model calls."""
    scorer_messages = [
        {"role": "system", "content": scorer_system},
        {"role": "user", "content": format_v5_scorer_user_message(prompt, essay)},
    ]
    scores = parse_and_normalize_scores(generate("scorer", scorer_messages, 96))
    feedback_messages = [
        {"role": "system", "content": feedback_system},
        {"role": "user", "content": format_v5_feedback_user_message(prompt, essay, scores)},
    ]
    try:
        feedback = parse_feedback(generate("feedback", feedback_messages, 256))
    except (ValueError, TypeError, json.JSONDecodeError):
        feedback = fallback_feedback(scores)
    return {"scores": scores, "total": compute_total(scores), "feedback": feedback}


def parse_and_normalize_scores(text: str) -> dict[str, int]:
    payload = _first_json_object(text)
    source = payload.get("scores")
    if not isinstance(source, Mapping):
        raise ValueError("Scorer response is missing scores")
    scores: dict[str, int] = {}
    for criterion in CRITERIA:
        value = source.get(criterion)
        if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Invalid scorer value for {criterion}")
        low, high = SCORE_RANGES[criterion]
        scores[criterion] = min(max(value, low), high)
    return scores


def parse_feedback(text: str) -> dict[str, str]:
    payload = _first_json_object(text)
    source = payload.get("feedback")
    if not isinstance(source, Mapping):
        raise ValueError("Feedback response is missing feedback")
    result: dict[str, str] = {}
    for criterion in CRITERIA:
        value = source.get(criterion)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid feedback for {criterion}")
        result[criterion] = value.strip()
    return result


def fallback_feedback(scores: Mapping[str, int]) -> dict[str, str]:
    return {
        criterion: (
            f"The validated {criterion.replace('_', ' ')} score is {scores[criterion]}; "
            "criterion-specific model feedback was unavailable."
        )
        for criterion in CRITERIA
    }


def _first_json_object(text: str) -> dict[str, Any]:
    for candidate, _, _ in extract_balanced_json_objects(text.strip()):
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value
