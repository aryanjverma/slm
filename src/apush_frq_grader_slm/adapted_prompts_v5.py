"""Deterministic adapted LEQ prompt families for v5 (no verbatim CB copies)."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from apush_frq_grader_slm.ingest.dedup import normalize_essay

_YEAR_RE = re.compile(
    r"\b((?:1[4-9]\d{2}|20[0-2]\d)(?:\s*[-–]\s*(?:1[4-9]\d{2}|20[0-2]\d))?)\b"
)

# Skill-preserving openers that are not the College Board stock phrasing.
_SKILL_OPENERS: dict[str, tuple[str, ...]] = {
    "extent": (
        "Assess how far",
        "Judge the degree to which",
        "Determine whether",
    ),
    "causation": (
        "Explain the ways",
        "Trace how",
        "Analyze the process by which",
    ),
    "ccot": (
        "Trace continuity and change as",
        "Assess patterns of continuity and change in how",
        "Examine how",
    ),
    "continuity_change": (
        "Trace continuity and change as",
        "Assess patterns of continuity and change in how",
        "Examine how",
    ),
    "relative_importance": (
        "Weigh the leading factors behind",
        "Rank the main causes of",
        "Compare the force of competing causes for",
    ),
    "comparison": (
        "Compare developments in",
        "Contrast patterns within",
        "Set side by side the trajectories of",
    ),
}

_STOCK_OPENERS = (
    re.compile(
        r"^Evaluate the extent to which\s+",
        re.I,
    ),
    re.compile(
        r"^Evaluate the relative importance of (?:the )?(?:causes|effects) of\s+",
        re.I,
    ),
    re.compile(
        r"^Evaluate how\s+",
        re.I,
    ),
    re.compile(
        r"^Evaluate\s+",
        re.I,
    ),
)

_FOCUS_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgrowth of transatlantic trade\b", re.I), "expansion of Atlantic commerce"),
    (re.compile(r"\bchanges in colonial societies in North America\b", re.I),
     "shifts inside North American colonial communities"),
    (re.compile(r"\bchanges in United States foreign policy\b", re.I),
     "reorientations in U.S. diplomacy"),
    (re.compile(r"\bchanges in debates over the role of the federal government\b", re.I),
     "shifting arguments about national authority"),
    (re.compile(r"\bgrowing concerns about national security\b", re.I),
     "rising national-security anxieties"),
    (re.compile(r"\bgrowth of civil rights activism\b", re.I), "surge of civil rights organizing"),
    (re.compile(r"\bcauses of conflict among Europeans and Native Americans\b", re.I),
     "sources of conflict between Europeans and Indigenous nations"),
    (re.compile(r"\bsettler expansion\b", re.I), "settler territorial push"),
    (re.compile(r"\bcauses of the growth of a national culture\b", re.I),
     "drivers behind a shared national culture"),
    (re.compile(r"\bmigration\b", re.I), "population movement"),
    (re.compile(r"\bmovements for social change\b", re.I), "campaigns for social reform"),
    (re.compile(r"\bNative American societies\b", re.I), "Indigenous societies"),
    (re.compile(r"\bBritish colonists in the Americas\b", re.I), "British settlers in the Americas"),
    (re.compile(r"\bdifferent reform movements in the United States\b", re.I),
     "varied U.S. reform movements"),
    (re.compile(r"\bsectional tensions\b", re.I), "sectional rivalries"),
    (re.compile(r"\bUnited States foreign policy\b", re.I), "U.S. foreign policy"),
    (re.compile(r"\bUnited States society\b", re.I), "U.S. society"),
    (re.compile(r"\bBritish North American colonial society\b", re.I),
     "colonial society in British North America"),
    (re.compile(r"\bwestern United States\b", re.I), "the western U.S."),
    (re.compile(r"\bfederal government action\b", re.I), "action by the national government"),
    (re.compile(r"\bindustrialization\b", re.I), "industrial growth"),
)

_VERB_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bchanged\b", re.I), "remade"),
    (re.compile(r"\bcontributed to\b", re.I), "helped drive"),
    (re.compile(r"\binfluenced\b", re.I), "reshaped"),
    (re.compile(r"\bshaped\b", re.I), "molded"),
    (re.compile(r"\bresponded to\b", re.I), "reacted to"),
    (re.compile(r"\badapted to\b", re.I), "adjusted to"),
)


def year_tokens(prompt: str) -> list[str]:
    """Return year / year-range tokens preserved from an official prompt."""
    return [m.replace("–", "-") for m in _YEAR_RE.findall(prompt)]


def _strip_stock_opener(prompt: str) -> str:
    text = prompt.strip()
    for pattern in _STOCK_OPENERS:
        updated = pattern.sub("", text, count=1)
        if updated != text:
            return updated.strip()
    return text


def _rewrite_focus(body: str, *, variant: int) -> str:
    text = body
    applied = 0
    for pattern, replacement in _FOCUS_REWRITES:
        if pattern.search(text):
            text = pattern.sub(replacement, text, count=1)
            applied += 1
            if applied > variant:
                break
    for index, (pattern, replacement) in enumerate(_VERB_REWRITES):
        if index % 3 == variant % 3 and pattern.search(text):
            text = pattern.sub(replacement, text, count=1)
            break
    # Light lexical churn so even unmatched prompts still diverge.
    churn = (
        (re.compile(r"\bthe United States\b"), "the U.S."),
        (re.compile(r"\bUnited States\b"), "U.S."),
        (re.compile(r"\bNorth America\b"), "North American lands"),
        (re.compile(r"\bsociety\b", re.I), "social order"),
    )
    for index, (pattern, replacement) in enumerate(churn):
        if index % 3 == variant % 3:
            text = pattern.sub(replacement, text, count=1)
    return re.sub(r"\s+", " ", text).strip(" .,")


def adapt_official_prompt(
    prompt: str,
    reasoning_skill: str = "",
    *,
    count: int = 3,
) -> list[str]:
    """Produce 2–3 adapted LEQ prompts that keep years/skill but rephrase focus.

    Adapted wording must differ from the official prompt (normalized) while
    retaining every year / year-range token found in the source.
    """
    official = str(prompt or "").strip()
    if not official:
        raise ValueError("official prompt is required")
    n = max(2, min(3, int(count)))
    skill_key = (reasoning_skill or "extent").strip().lower()
    openers = _SKILL_OPENERS.get(skill_key) or _SKILL_OPENERS["extent"]
    body = _strip_stock_opener(official)
    years = year_tokens(official)
    adapted: list[str] = []
    for index in range(n):
        opener = openers[index % len(openers)]
        focus = _rewrite_focus(body, variant=index)
        candidate = f"{opener} {focus}".strip()
        if not candidate.endswith("."):
            candidate += "."
        # Ensure year tokens survive even if a rewrite clipped them.
        for year in years:
            if year not in candidate and year.replace("-", "–") not in candidate:
                candidate = candidate.rstrip(".") + f" ({year})."
        if normalize_essay(candidate) == normalize_essay(official):
            candidate = f"{opener} {_rewrite_focus(body + ' patterns', variant=index + 1)}."
        if normalize_essay(candidate) != normalize_essay(official):
            adapted.append(candidate)
    # Guarantee at least two distinct adaptations.
    while len(adapted) < 2:
        extra = (
            f"{openers[len(adapted) % len(openers)]} "
            f"{_rewrite_focus(body, variant=len(adapted) + 5)}."
        )
        if normalize_essay(extra) != normalize_essay(official):
            adapted.append(extra)
        else:
            adapted.append(extra.rstrip(".") + " in this era.")
    return adapted[:n]


def build_adapted_prompt_family_row(
    family: Mapping[str, Any],
    *,
    count: int = 3,
) -> dict[str, Any]:
    """Attach adapted prompts to one prompt-family catalog row."""
    prompt = str(family.get("prompt") or family.get("prompt_text") or "").strip()
    skill = str(family.get("reasoning_skill") or "")
    adapted = adapt_official_prompt(prompt, skill, count=count)
    row = dict(family)
    row["adapted_prompts"] = adapted
    row["official_prompt"] = prompt
    return row


def attach_adapted_prompts_to_seeds(
    seeds: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy family-level adapted prompts onto matching seed profiles."""
    by_family: dict[str, list[str]] = {}
    for family in families:
        family_id = str(family.get("prompt_family_id") or "")
        adapted = [str(item) for item in (family.get("adapted_prompts") or ()) if str(item).strip()]
        if family_id and adapted:
            by_family[family_id] = adapted
    updated: list[dict[str, Any]] = []
    for seed in seeds:
        row = dict(seed)
        family_id = str(row.get("prompt_family_id") or "")
        if family_id in by_family:
            row["adapted_prompts"] = list(by_family[family_id])
        elif not row.get("adapted_prompts"):
            prompt = str(row.get("prompt") or "")
            skill = str(row.get("reasoning_skill") or "")
            if prompt:
                row["adapted_prompts"] = adapt_official_prompt(prompt, skill)
        updated.append(row)
    return updated
