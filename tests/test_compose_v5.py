"""Focused tests for the score-blind v5 essay composer."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from apush_frq_grader_slm.compose_v4 import rng_for_task
from apush_frq_grader_slm.compose_v5 import (
    GENERATOR_NAME,
    _evidence_label,
    _topic_phrase,
    compose_essay,
    resolve_observable_behavior,
)
from apush_frq_grader_slm.fact_cards_v5 import has_eight_gram_overlap
from apush_frq_grader_slm.judge_v5 import _has_repeated_long_span


def _packet(**overrides: object) -> dict:
    base: dict = {
        "task_id": "v5-test-0001",
        "prompt": (
            "Assess how far the expansion of Atlantic commerce remade "
            "British North American colonial social order from 1607 to 1776."
        ),
        "student_capability": {
            "historical_knowledge": "competent",
            "argument_control": "consistent",
        },
        "timed_composition_style": {
            "time_pressure": "normal",
            "mechanics": "minor_errors",
            "organization": "clear",
        },
        "style_reference": "",
        "semantic_fact_cards": [
            {
                "concept": (
                    "Transatlantic trade tied colonial ports to British credit and "
                    "consumer goods across the eighteenth century."
                ),
                "use": "paraphrase from memory; do not copy wording",
            },
            {
                "concept": (
                    "Plantation slavery expanded as Atlantic markets rewarded "
                    "staple crops in the southern colonies."
                ),
                "use": "paraphrase from memory; do not copy wording",
            },
            {
                "concept": (
                    "Northern shipbuilding and merchant networks grew alongside "
                    "the same Atlantic commercial circuit."
                ),
                "use": "paraphrase from memory; do not copy wording",
            },
        ],
    }
    base.update(overrides)
    return base


class ComposeV5Tests(unittest.TestCase):
    def test_limited_knowledge_shorter_and_weaker_than_strong(self) -> None:
        limited = _packet(
            task_id="v5-test-limited",
            student_capability={
                "historical_knowledge": "limited",
                "argument_control": "emerging",
            },
            timed_composition_style={
                "time_pressure": "severe",
                "mechanics": "frequent_natural_errors",
                "organization": "rough",
            },
        )
        strong = _packet(
            task_id="v5-test-strong",
            student_capability={
                "historical_knowledge": "strong",
                "argument_control": "nuanced",
            },
            timed_composition_style={
                "time_pressure": "normal",
                "mechanics": "minor_errors",
                "organization": "clear",
            },
        )
        limited_essay = compose_essay(limited, rng=rng_for_task("v5-test-limited"))
        strong_essay = compose_essay(strong, rng=rng_for_task("v5-test-strong"))
        self.assertLess(
            len(limited_essay.split()),
            len(strong_essay.split()),
            msg=(
                f"expected limited essay shorter than strong; "
                f"limited={len(limited_essay.split())} strong={len(strong_essay.split())}"
            ),
        )
        # Weak / emerging control should avoid a crisp overall claim framing.
        strong_markers = (
            "to a significant extent",
            "overall,",
            "i think",
            "mattered",
        )
        self.assertTrue(
            any(m in strong_essay.lower() for m in strong_markers),
            msg=f"strong essay lacked claim-like framing:\n{strong_essay[:300]}",
        )
        self.assertFalse(
            "to a significant extent" in limited_essay.lower()
            and "mainly becuase" in limited_essay.lower(),
            msg=f"limited essay unexpectedly used strong claim template:\n{limited_essay[:300]}",
        )

    def test_no_score_keys_leaked_in_packet_or_contract(self) -> None:
        packet = _packet()
        essay = compose_essay(packet, rng=rng_for_task(packet["task_id"]))
        self.assertIsInstance(essay, str)
        self.assertGreater(len(essay.split()), 80)
        # Composer must refuse packets that already carry score targets.
        dirty = dict(packet)
        dirty["target_scores"] = {"thesis": 1}
        with self.assertRaises(ValueError):
            compose_essay(dirty, rng=rng_for_task("v5-test-leak"))
        # Output contract for shard rows stays score-blind.
        row = {
            "task_id": packet["task_id"],
            "student_response": essay,
            "generator_name": GENERATOR_NAME,
            "shard_id": "v5-shard-00",
        }
        forbidden = {
            "target_scores",
            "target_total",
            "scores",
            "score",
            "rubric_text",
            "resolved_grade",
        }
        self.assertFalse(forbidden & set(row))
        self.assertEqual(row["generator_name"], "compose_v5_score_blind")

    def test_essay_references_paraphrased_concepts_when_cards_present(self) -> None:
        packet = _packet(
            task_id="v5-test-cards",
            student_capability={
                "historical_knowledge": "strong",
                "argument_control": "nuanced",
            },
        )
        essay = compose_essay(packet, rng=rng_for_task("v5-test-cards"))
        lowered = essay.lower()
        # Expect paraphrased anchors drawn from card concepts (not AMSCO copy).
        anchors = (
            "trade",
            "atlantic",
            "slavery",
            "plantation",
            "merchant",
            "ship",
            "colonial",
            "staple",
        )
        hits = [a for a in anchors if a in lowered]
        self.assertGreaterEqual(
            len(hits),
            2,
            msg=f"expected paraphrased concept cues in essay; hits={hits}\n{essay}",
        )
        for card in packet["semantic_fact_cards"]:
            concept = card["concept"]
            self.assertFalse(
                has_eight_gram_overlap(essay, concept),
                msg=f"essay copied 8-gram from concept:\n{concept}\n---\n{essay}",
            )

    def test_style_reference_not_copied_as_long_verbatim_span(self) -> None:
        style = (
            "Although it furthered the British North American colonies ties to England, "
            "transatlantic trade succeeded in drastically changing the British North "
            "American colonial society to large extent by leading to the development of "
            "distinct regions throughout the colonies and leading to the dependence of "
            "Southern society on slave labor."
        )
        packet = _packet(
            task_id="v5-test-style",
            style_reference=style[:400],
            student_capability={
                "historical_knowledge": "competent",
                "argument_control": "consistent",
            },
        )
        essay = compose_essay(packet, rng=rng_for_task("v5-test-style"))
        self.assertFalse(
            has_eight_gram_overlap(essay, style),
            msg=f"essay retained 8-gram overlap with style_reference:\n{essay}",
        )
        # Long contiguous clone of the style opening should not appear.
        self.assertNotIn(
            "transatlantic trade succeeded in drastically changing",
            essay.lower(),
        )

    def test_boundary_behavior_restored_from_task(self) -> None:
        packet = _packet(
            task_id="v5-test-boundary",
            student_capability={
                "historical_knowledge": "uneven",
                "argument_control": "partial",
            },
        )
        task = {
            "task_id": "v5-test-boundary",
            "capability_profile": {
                "historical_knowledge": "uneven",
                "argument_control": "partial",
                "observable_writing_behavior": (
                    "discusses the topic but never settles on one overall answer"
                ),
            },
        }
        behavior = resolve_observable_behavior(packet, task)
        self.assertIn("never settles", behavior or "")
        essay = compose_essay(
            packet,
            observable_writing_behavior=behavior,
            rng=rng_for_task("v5-test-boundary"),
        )
        lowered = essay.lower()
        self.assertTrue(
            any(
                cue in lowered
                for cue in (
                    "hard to pick one answer",
                    "not sure there is one clear",
                    "different directions",
                    "lots of factors",
                )
            ),
            msg=f"boundary unsettle thesis not reflected:\n{essay}",
        )

    def test_deterministic_for_same_rng(self) -> None:
        packet = _packet(task_id="v5-test-det")
        a = compose_essay(packet, rng=rng_for_task("v5-test-det"))
        b = compose_essay(packet, rng=rng_for_task("v5-test-det"))
        self.assertEqual(a, b)

    def test_topic_phrase_stays_short_noun_focus(self) -> None:
        prompt = (
            "Explain the ways sectional rivalries molded United States social "
            "order from 1800 to 1848."
        )
        topic = _topic_phrase(prompt)
        self.assertLessEqual(len(topic.split()), 10)
        self.assertNotRegex(topic.lower(), r"^(explain|evaluate|assess|analyze)\b")
        self.assertIn("sectional rivalries", topic.lower())
        self.assertRegex(topic, r"1800-1848")

    def test_evidence_label_rejects_bare_truncated_names(self) -> None:
        rng = rng_for_task("v5-test-evidence-label")
        rejected = (
            "Prelude is a remembered reference point in period 8 (1945).",
            "September is a remembered reference point in period 8 (1939).",
            "Cotton is a remembered reference point in period 4 (1830).",
            "Allies is a remembered reference point in period 7 (1917).",
            "Few is a remembered reference point in period 4 (1800).",
        )
        for concept in rejected:
            label = _evidence_label(concept, rng)
            self.assertNotRegex(
                label,
                r"^(Prelude|September|Cotton|Allies|Few)$",
                msg=f"bare truncated label from {concept!r}: {label!r}",
            )
            self.assertTrue(2 <= len(label.split()) <= 8, msg=label)

        good = _evidence_label(
            "Emancipation Proclamation is a remembered reference point in period 5 (1863).",
            rng,
        )
        self.assertIn("Emancipation Proclamation", good)
        battle = _evidence_label(
            "Battle of New Orleans is a remembered reference point in period 4 (1815).",
            rng,
        )
        self.assertIn("Battle of New Orleans", battle)

    def test_quality_hygiene_across_random_seeds(self) -> None:
        prompts = [
            (
                "Explain the ways sectional rivalries molded United States social "
                "order from 1800 to 1848."
            ),
            (
                "Assess how far the expansion of Atlantic commerce remade "
                "British North American colonial social order from 1607 to 1776."
            ),
            (
                "Evaluate the extent to which the Spanish-American War changed "
                "United States foreign policy from 1898 to 1914."
            ),
        ]
        messy_cards = [
            {"concept": "The Old Northwest is a remembered reference point in period 4 (1860)."},
            {"concept": "By is a remembered reference point in period 4 (1861)."},
            {"concept": "The is a remembered reference point in period 4 (1787)."},
            {"concept": "Alger Hiss is a remembered reference point in period 8 (1948)."},
            {
                "concept": (
                    "Treaty of Tordesillas renegotiated the papal demarcation and "
                    "formalized it as a lasting Iberian settlement in 1494."
                )
            },
            {
                "concept": (
                    "Transatlantic trade tied colonial ports to British credit and "
                    "consumer goods across the eighteenth century."
                )
            },
        ]
        for seed in range(20):
            prompt = prompts[seed % len(prompts)]
            topic = _topic_phrase(prompt)
            self.assertLessEqual(
                len(topic.split()),
                10,
                msg=f"topic too long for seed {seed}: {topic!r}",
            )
            packet = _packet(
                task_id=f"v5-test-hygiene-{seed:02d}",
                prompt=prompt,
                semantic_fact_cards=messy_cards,
                student_capability={
                    "historical_knowledge": ("competent", "strong", "uneven", "limited")[
                        seed % 4
                    ],
                    "argument_control": ("consistent", "partial", "emerging", "nuanced")[
                        seed % 4
                    ],
                },
                timed_composition_style={
                    "time_pressure": ("normal", "severe", "moderate")[seed % 3],
                    "mechanics": (
                        "minor_errors",
                        "frequent_natural_errors",
                        "fragments_and_runons",
                        "occasional_errors",
                    )[seed % 4],
                    "organization": ("clear", "rough", "uneven", "repetitive")[seed % 4],
                },
            )
            essay = compose_essay(packet, rng=rng_for_task(packet["task_id"]))
            self.assertIsNone(
                re.search(r"\bmattering\b", essay, flags=re.I),
                msg=f"mattering stub in seed {seed}:\n{essay[:400]}",
            )
            prompt_words = prompt.split()
            lowered = essay.lower()
            for i in range(max(0, len(prompt_words) - 11)):
                span = " ".join(prompt_words[i : i + 12]).lower()
                self.assertLess(
                    lowered.count(span),
                    3,
                    msg=(
                        f"12-word prompt span repeated ≥3 times (seed {seed}): "
                        f"{span!r}\n{essay[:400]}"
                    ),
                )

    def test_real_packets_avoid_repeated_10grams(self) -> None:
        packets_path = Path("artifacts/data/v5/packets/v5-shard-00.jsonl")
        tasks_path = Path("artifacts/data/v5/planning/generation_tasks_v5.jsonl")
        if not packets_path.is_file():
            self.skipTest(f"missing packets shard: {packets_path}")
        packets = [
            json.loads(line)
            for line in packets_path.read_text().splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(packets), 30)
        tasks: dict[str, dict] = {}
        if tasks_path.is_file():
            for line in tasks_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("task_id"):
                    tasks[str(row["task_id"])] = row

        essays: list[str] = []
        for packet in packets[:30]:
            task_id = str(packet["task_id"])
            behavior = resolve_observable_behavior(packet, tasks.get(task_id))
            essay = compose_essay(
                packet,
                observable_writing_behavior=behavior,
                rng=rng_for_task(task_id),
            )
            essays.append(essay)
            self.assertIsNone(
                re.search(r"\bmattering\b", essay, flags=re.I),
                msg=f"mattering stub in {task_id}:\n{essay[:300]}",
            )

        repeated = sum(1 for essay in essays if _has_repeated_long_span(essay))
        self.assertLess(
            repeated / len(essays),
            0.20,
            msg=f"{repeated}/{len(essays)} essays had a 10-word span repeated ≥3 times",
        )
        mean_words = sum(len(essay.split()) for essay in essays) / len(essays)
        self.assertGreaterEqual(
            mean_words,
            140,
            msg=f"mean word count {mean_words:.1f} below 140",
        )


if __name__ == "__main__":
    unittest.main()
