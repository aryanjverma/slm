import unittest

from apush_frq_grader_slm.baselines import InflatedPromptedBase, ReferenceGrader
from apush_frq_grader_slm.data import generate_cases
from apush_frq_grader_slm.eval import evaluate_adapter, summarize
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


if __name__ == "__main__":
    unittest.main()
