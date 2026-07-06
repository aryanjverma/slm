import unittest

from arithmetic_tutor_slm.baselines import LeakyPromptedBase, SocraticTutorReference
from arithmetic_tutor_slm.data import generate_cases
from arithmetic_tutor_slm.eval import evaluate_adapter, summarize
from arithmetic_tutor_slm.filters import passes_quality_gate


class CoreBehaviorTests(unittest.TestCase):
    def test_generated_cases_pass_quality_gate(self):
        cases = generate_cases(count=25, split="train", seed=7)
        failures = [passes_quality_gate(case) for case in cases]
        self.assertTrue(all(ok for ok, _ in failures))

    def test_eval_separates_answer_leak_from_tutor_behavior(self):
        cases = generate_cases(count=25, split="eval", seed=8, adversarial_ratio=0.3)
        leaky = summarize(evaluate_adapter(cases, LeakyPromptedBase()), "leaky")
        tutor = summarize(evaluate_adapter(cases, SocraticTutorReference()), "tutor")

        self.assertEqual(leaky.no_answer_leak_rate, 0)
        self.assertEqual(tutor.no_answer_leak_rate, 1)
        self.assertGreater(tutor.total_score_mean, leaky.total_score_mean)


if __name__ == "__main__":
    unittest.main()
