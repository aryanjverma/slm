"""Knowledge bases for APUSH grading / synthetic data enrichment."""

from apush_frq_grader_slm.knowledge.amsco import (
    DEFAULT_KB_PATH,
    facts_for_period,
    facts_for_prompt,
    load_kb,
)

__all__ = [
    "DEFAULT_KB_PATH",
    "facts_for_period",
    "facts_for_prompt",
    "load_kb",
]
