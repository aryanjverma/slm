import unittest

from apush_frq_grader_slm.baselines import InflatedPromptedBase, ReferenceGrader
from apush_frq_grader_slm.data import generate_cases
from apush_frq_grader_slm.eval import (
    _quadratic_weighted_kappa,
    evaluate_adapter,
    summarize,
    summarize_by_dimensions,
    summarize_real_eval_by_rubric,
)
from apush_frq_grader_slm.filters import passes_quality_gate, parse_grade_json
from apush_frq_grader_slm.rubric import validate_grade_payload


class CoreBehaviorTests(unittest.TestCase):
    def test_generated_cases_pass_quality_gate(self):
        cases = generate_cases(count=25, split="train", seed=7)
        failures = [passes_quality_gate(case) for case in cases]
        self.assertTrue(all(ok for ok, _ in failures))

    def test_reference_grade_json_is_valid(self):
        cases = generate_cases(count=5, split="train", seed=3)
        for case in cases:
            payload, reasons = parse_grade_json(case.assistant_response)
            self.assertIsNotNone(payload, msg=str(reasons))
            ok, validation_reasons = validate_grade_payload(payload)
            self.assertTrue(ok, msg=str(validation_reasons))

    def test_eval_separates_inflated_base_from_reference_grader(self):
        cases = generate_cases(count=25, split="eval", seed=8, adversarial_ratio=0.3)
        inflated = summarize(evaluate_adapter(cases, InflatedPromptedBase()), "inflated")
        reference = summarize(evaluate_adapter(cases, ReferenceGrader()), "reference")

        self.assertLess(inflated.rubric_accuracy_mean, reference.rubric_accuracy_mean)
        self.assertLess(inflated.evidence_grounding_rate, reference.evidence_grounding_rate)
        self.assertGreater(reference.total_score_mean, inflated.total_score_mean)

    def test_qwk_handles_totals_outside_rubric_range(self):
        # A model can emit a total outside 0-6; QWK must clamp, not IndexError.
        ref = [4, 3, 5, 2, 6, 4]
        pred = [7, 3, 8, 2, 6, 4]  # 7 and 8 previously crashed the confusion matrix
        kappa = _quadratic_weighted_kappa(ref, pred)
        self.assertIsNotNone(kappa)
        self.assertLessEqual(kappa, 1.0)
        # Perfect agreement still scores 1.0 after the clamp.
        self.assertEqual(_quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0)

    def test_real_eval_splits_rubric_versions(self):
        cases = generate_cases(count=2, split="eval", seed=4)
        cases[0].provenance.year = 2023
        cases[0].provenance.rubric_version = "2023_leq"
        cases[1].provenance.year = 2025
        cases[1].provenance.rubric_version = "2024_2026_leq"
        results = evaluate_adapter(cases, ReferenceGrader())
        summaries = summarize_real_eval_by_rubric(results, cases)
        self.assertEqual(set(summaries), {"2023_leq", "2024_2026_leq"})
        self.assertTrue(all(summary.total_mae == 0 for summary in summaries.values()))

    def test_dimension_summary_includes_v2_operational_slices(self):
        cases = generate_cases(count=3, split="eval", seed=5)
        cases[0].provenance.prompt_family_id = "family_a"
        cases[0].provenance.generator_config = {
            "period": 4,
            "reasoning_skill": "causation",
            "persona": {"time_budget_minutes": 30, "historical_knowledge": "competent"},
        }
        results = evaluate_adapter(cases, ReferenceGrader())
        dimensions = summarize_by_dimensions(results, cases)
        self.assertIn("family_a", dimensions["prompt_family"])
        self.assertIn("30", dimensions["time_budget"])
        self.assertIn("4", dimensions["period"])


if __name__ == "__main__":
    unittest.main()
