"""Original APUSH LEQ prompt families, deterministic splits, and leakage checks."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReasoningSkill(StrEnum):
    CAUSATION = "causation"
    COMPARISON = "comparison"
    CONTINUITY_CHANGE = "continuity_change"


class PromptSourceType(StrEnum):
    ORIGINAL = "original"
    ADAPTED = "adapted"
    OFFICIAL = "official"


class PromptFormat(StrEnum):
    STANDARD = "standard_leq"
    MAY_2027_BROAD = "may_2027_broad"


class PromptSplit(StrEnum):
    TRAIN = "train"
    SYNTHETIC_DEV = "synthetic_dev"
    SYNTHETIC_CHALLENGE = "synthetic_challenge"


DEFAULT_SPLIT_RATIOS: dict[PromptSplit, float] = {
    PromptSplit.TRAIN: 0.70,
    PromptSplit.SYNTHETIC_DEV: 0.15,
    PromptSplit.SYNTHETIC_CHALLENGE: 0.15,
}
DEFAULT_SPLIT_SEED = "apush-original-prompt-bank-v1"
DEFAULT_HARD_SIMILARITY_THRESHOLD = 0.72
DEFAULT_REVIEW_SIMILARITY_THRESHOLD = 0.52


class PromptDateRange(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_year: int = Field(ge=1491, le=2100)
    end_year: int = Field(ge=1491, le=2100)

    @model_validator(mode="after")
    def _ordered(self) -> PromptDateRange:
        if self.end_year < self.start_year:
            raise ValueError("end_year must not precede start_year")
        return self


class PromptFamily(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    prompt_family_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    prompt_text: str = Field(min_length=40)
    source_prompt_text: str = Field(min_length=40)
    source_name: str = Field(min_length=3)
    source_year: int = Field(ge=1900, le=2100)
    source_type: PromptSourceType
    parent_prompt: str | None = None
    period: int = Field(ge=2, le=9)
    reasoning_skill: ReasoningSkill
    date_range: PromptDateRange
    topic: str = Field(min_length=5)
    valid_evidence: tuple[str, ...] = Field(min_length=3)
    common_misconceptions: tuple[str, ...] = Field(min_length=2)
    format_version: PromptFormat = PromptFormat.STANDARD
    is_2027_challenge: bool = False

    @model_validator(mode="after")
    def _validate_provenance_and_format(self) -> PromptFamily:
        if self.source_type == PromptSourceType.ORIGINAL:
            if self.parent_prompt is not None:
                raise ValueError("original prompts cannot have a parent_prompt")
            if self.source_prompt_text != self.prompt_text:
                raise ValueError("source_prompt_text must equal prompt_text for original prompts")
        if self.source_type == PromptSourceType.ADAPTED and not self.parent_prompt:
            raise ValueError("adapted prompts require the exact parent_prompt")
        if self.is_2027_challenge != (self.format_version == PromptFormat.MAY_2027_BROAD):
            raise ValueError("2027 challenge marker and format_version must agree")
        return self


class PromptCatalogEntry(PromptFamily):
    catalog_version: Literal["original_v1"] = "original_v1"
    split_seed: str = DEFAULT_SPLIT_SEED
    split: PromptSplit


class CatalogValidationIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    severity: Literal["error", "review"]
    message: str
    family_ids: tuple[str, ...] = ()
    similarity: float | None = None


class CatalogValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    issues: tuple[CatalogValidationIssue, ...]
    family_count: int
    prompt_count: int
    split_counts: dict[PromptSplit, int]
    max_cross_split_similarity: float

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def raise_for_issues(self) -> None:
        if self.is_valid:
            return
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)
        raise ValueError(f"Prompt catalog validation failed: {details}")


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_SIMILARITY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "argument",
        "as",
        "at",
        "between",
        "by",
        "change",
        "changes",
        "continuities",
        "continuity",
        "develop",
        "developments",
        "did",
        "different",
        "during",
        "evaluate",
        "explaining",
        "extent",
        "for",
        "from",
        "historical",
        "how",
        "in",
        "most",
        "of",
        "over",
        "period",
        "selected",
        "shaped",
        "similar",
        "that",
        "the",
        "their",
        "these",
        "through",
        "to",
        "using",
        "ways",
        "which",
        "with",
        "your",
    }
)


def _similarity_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _TOKEN_PATTERN.findall(text.casefold())
        if token not in _SIMILARITY_STOPWORDS
    )


def prompt_token_similarity(left: str, right: str) -> float:
    """Return Jaccard token similarity after removing LEQ boilerplate words."""
    left_tokens = _similarity_tokens(left)
    right_tokens = _similarity_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _normalized_text(text: str) -> str:
    return " ".join(_TOKEN_PATTERN.findall(text.casefold()))


def _ranges_overlap(left: PromptDateRange, right: PromptDateRange) -> bool:
    return left.start_year <= right.end_year and right.start_year <= left.end_year


def _stable_key(seed: str, family_id: str, purpose: str) -> str:
    payload = f"{seed}:{purpose}:{family_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_family_counts(
    family_count: int,
    ratios: Mapping[PromptSplit, float] = DEFAULT_SPLIT_RATIOS,
) -> dict[PromptSplit, int]:
    """Allocate exact family counts with deterministic largest remainders."""
    if family_count < 1:
        raise ValueError("family_count must be positive")
    if set(ratios) != set(PromptSplit):
        raise ValueError("ratios must define every PromptSplit exactly once")
    if any(value < 0 for value in ratios.values()) or not math.isclose(
        sum(ratios.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("split ratios must be nonnegative and sum to 1")

    exact = {split: family_count * ratios[split] for split in PromptSplit}
    counts = {split: math.floor(exact[split]) for split in PromptSplit}
    remainder = family_count - sum(counts.values())
    priority = {split: index for index, split in enumerate(PromptSplit)}
    ranked = sorted(
        PromptSplit,
        key=lambda split: (-(exact[split] - counts[split]), priority[split]),
    )
    for split in ranked[:remainder]:
        counts[split] += 1
    return counts


def _balanced_family_selection(
    candidates: Sequence[PromptFamily],
    count: int,
    *,
    seed: str,
    purpose: str,
    existing: Sequence[PromptFamily] = (),
) -> list[PromptFamily]:
    selected = list(existing)
    remaining = list(candidates)
    additions: list[PromptFamily] = []
    while len(additions) < count:
        period_counts = Counter(item.period for item in selected)
        skill_counts = Counter(item.reasoning_skill for item in selected)
        pair_counts = Counter((item.period, item.reasoning_skill) for item in selected)
        chosen = min(
            remaining,
            key=lambda item: (
                period_counts[item.period],
                skill_counts[item.reasoning_skill],
                pair_counts[(item.period, item.reasoning_skill)],
                _stable_key(seed, item.prompt_family_id, purpose),
            ),
        )
        additions.append(chosen)
        selected.append(chosen)
        remaining.remove(chosen)
    return additions


def assign_family_splits(
    prompts: Sequence[PromptFamily],
    *,
    seed: str = DEFAULT_SPLIT_SEED,
    ratios: Mapping[PromptSplit, float] = DEFAULT_SPLIT_RATIOS,
) -> list[PromptCatalogEntry]:
    """Assign whole families to exact, deterministic, coverage-aware splits."""
    if not prompts:
        raise ValueError("at least one prompt is required")

    grouped: dict[str, list[PromptFamily]] = defaultdict(list)
    for prompt in prompts:
        grouped[prompt.prompt_family_id].append(prompt)
    representatives = [
        sorted(rows, key=lambda row: row.prompt_id)[0] for rows in grouped.values()
    ]
    targets = split_family_counts(len(representatives), ratios)

    reserved_challenge = sorted(
        (item for item in representatives if item.is_2027_challenge),
        key=lambda item: item.prompt_family_id,
    )
    if len(reserved_challenge) > targets[PromptSplit.SYNTHETIC_CHALLENGE]:
        raise ValueError("2027 challenge families exceed the synthetic challenge allocation")

    reserved_ids = {item.prompt_family_id for item in reserved_challenge}
    remaining = [item for item in representatives if item.prompt_family_id not in reserved_ids]
    challenge_additions = _balanced_family_selection(
        remaining,
        targets[PromptSplit.SYNTHETIC_CHALLENGE] - len(reserved_challenge),
        seed=seed,
        purpose=PromptSplit.SYNTHETIC_CHALLENGE.value,
        existing=reserved_challenge,
    )
    challenge = [*reserved_challenge, *challenge_additions]
    challenge_ids = {item.prompt_family_id for item in challenge}
    remaining = [item for item in remaining if item.prompt_family_id not in challenge_ids]
    dev = _balanced_family_selection(
        remaining,
        targets[PromptSplit.SYNTHETIC_DEV],
        seed=seed,
        purpose=PromptSplit.SYNTHETIC_DEV.value,
    )
    dev_ids = {item.prompt_family_id for item in dev}

    split_by_family = {
        **{item.prompt_family_id: PromptSplit.SYNTHETIC_CHALLENGE for item in challenge},
        **{item.prompt_family_id: PromptSplit.SYNTHETIC_DEV for item in dev},
        **{
            item.prompt_family_id: PromptSplit.TRAIN
            for item in representatives
            if item.prompt_family_id not in challenge_ids | dev_ids
        },
    }
    return [
        PromptCatalogEntry.model_validate(
            {
                **prompt.model_dump(mode="python"),
                "split": split_by_family[prompt.prompt_family_id],
                "split_seed": seed,
            }
        )
        for prompt in sorted(prompts, key=lambda item: item.prompt_id)
    ]


def _review_pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def validate_prompt_catalog(
    entries: Sequence[PromptCatalogEntry],
    *,
    protected_prompts: Iterable[str] = (),
    hard_similarity_threshold: float = DEFAULT_HARD_SIMILARITY_THRESHOLD,
    review_similarity_threshold: float = DEFAULT_REVIEW_SIMILARITY_THRESHOLD,
    reviewed_pairs: Iterable[tuple[str, str]] = (),
    enforce_catalog_requirements: bool = True,
    ratios: Mapping[PromptSplit, float] = DEFAULT_SPLIT_RATIOS,
) -> CatalogValidationReport:
    """Validate coverage, provenance, split isolation, and prompt similarity."""
    if not 0 <= review_similarity_threshold < hard_similarity_threshold <= 1:
        raise ValueError("similarity thresholds must satisfy 0 <= review < hard <= 1")

    issues: list[CatalogValidationIssue] = []
    reviewed = {_review_pair_key(*pair) for pair in reviewed_pairs}
    ids = [entry.prompt_id for entry in entries]
    for prompt_id, count in Counter(ids).items():
        if count > 1:
            issues.append(
                CatalogValidationIssue(
                    code="duplicate_prompt_id",
                    severity="error",
                    message=f"prompt_id {prompt_id!r} appears {count} times",
                )
            )

    by_family: dict[str, list[PromptCatalogEntry]] = defaultdict(list)
    for entry in entries:
        by_family[entry.prompt_family_id].append(entry)
    for family_id, family_entries in by_family.items():
        splits = {entry.split for entry in family_entries}
        if len(splits) > 1:
            issues.append(
                CatalogValidationIssue(
                    code="family_split_leakage",
                    severity="error",
                    message=f"family {family_id!r} appears in multiple splits",
                    family_ids=(family_id,),
                )
            )
        signatures = {
            (
                entry.period,
                entry.reasoning_skill,
                entry.date_range.start_year,
                entry.date_range.end_year,
                _normalized_text(entry.topic),
            )
            for entry in family_entries
        }
        if len(signatures) > 1:
            issues.append(
                CatalogValidationIssue(
                    code="inconsistent_family_metadata",
                    severity="error",
                    message=f"family {family_id!r} has inconsistent scope metadata",
                    family_ids=(family_id,),
                )
            )

    family_split = {
        family_id: family_entries[0].split for family_id, family_entries in by_family.items()
    }
    actual_counts = Counter(family_split.values())
    split_counts = {split: actual_counts[split] for split in PromptSplit}

    if enforce_catalog_requirements:
        if len(by_family) < 60:
            issues.append(
                CatalogValidationIssue(
                    code="insufficient_family_count",
                    severity="error",
                    message=f"catalog has {len(by_family)} families; at least 60 are required",
                )
            )
        non_original = [
            entry.prompt_family_id
            for entry in entries
            if entry.source_type != PromptSourceType.ORIGINAL
        ]
        if non_original:
            issues.append(
                CatalogValidationIssue(
                    code="non_original_source",
                    severity="error",
                    message="the project prompt bank must contain only original prompts",
                    family_ids=tuple(sorted(set(non_original))),
                )
            )
        periods = {entry.period for entry in entries}
        if periods != set(range(2, 10)):
            issues.append(
                CatalogValidationIssue(
                    code="period_coverage",
                    severity="error",
                    message=f"period coverage is {sorted(periods)}; expected periods 2-9",
                )
            )
        for period in range(2, 10):
            period_skills = {
                entry.reasoning_skill for entry in entries if entry.period == period
            }
            if period_skills != set(ReasoningSkill):
                issues.append(
                    CatalogValidationIssue(
                        code="period_skill_coverage",
                        severity="error",
                        message=f"period {period} does not cover all reasoning skills",
                    )
                )
        expected_counts = split_family_counts(len(by_family), ratios)
        if split_counts != expected_counts:
            issues.append(
                CatalogValidationIssue(
                    code="split_ratio",
                    severity="error",
                    message=f"family split counts are {split_counts}; expected {expected_counts}",
                )
            )
        forward_families = {
            entry.prompt_family_id for entry in entries if entry.is_2027_challenge
        }
        if not forward_families:
            issues.append(
                CatalogValidationIssue(
                    code="missing_2027_challenge",
                    severity="error",
                    message="catalog must include at least one May 2027 format challenge family",
                )
            )
        if len(forward_families) > max(3, math.ceil(len(by_family) * 0.10)):
            issues.append(
                CatalogValidationIssue(
                    code="oversized_2027_challenge",
                    severity="error",
                    message="May 2027 format markers must remain a small challenge slice",
                    family_ids=tuple(sorted(forward_families)),
                )
            )
        misplaced = {
            entry.prompt_family_id
            for entry in entries
            if entry.is_2027_challenge and entry.split != PromptSplit.SYNTHETIC_CHALLENGE
        }
        if misplaced:
            issues.append(
                CatalogValidationIssue(
                    code="misplaced_2027_challenge",
                    severity="error",
                    message="May 2027 format families must be held in synthetic_challenge",
                    family_ids=tuple(sorted(misplaced)),
                )
            )

    max_cross_split_similarity = 0.0
    for index, left in enumerate(entries):
        for right in entries[index + 1 :]:
            similarity = prompt_token_similarity(left.prompt_text, right.prompt_text)
            if left.split != right.split:
                max_cross_split_similarity = max(max_cross_split_similarity, similarity)
            pair = _review_pair_key(left.prompt_family_id, right.prompt_family_id)
            exact_duplicate = _normalized_text(left.prompt_text) == _normalized_text(
                right.prompt_text
            )
            if exact_duplicate:
                issues.append(
                    CatalogValidationIssue(
                        code="duplicate_prompt_text",
                        severity="error",
                        message="two catalog prompts normalize to identical text",
                        family_ids=pair,
                        similarity=1.0,
                    )
                )
                continue
            if left.split != right.split and similarity >= hard_similarity_threshold:
                issues.append(
                    CatalogValidationIssue(
                        code="cross_split_near_paraphrase",
                        severity="error",
                        message="cross-split prompt similarity exceeds the hard threshold",
                        family_ids=pair,
                        similarity=similarity,
                    )
                )
            elif (
                left.split != right.split
                and similarity >= review_similarity_threshold
                and pair not in reviewed
            ):
                issues.append(
                    CatalogValidationIssue(
                        code="manual_similarity_review_required",
                        severity="review",
                        message="cross-split prompt pair requires topic/date-range review",
                        family_ids=pair,
                        similarity=similarity,
                    )
                )
            same_topic = _normalized_text(left.topic) == _normalized_text(right.topic)
            if (
                left.split != right.split
                and same_topic
                and _ranges_overlap(left.date_range, right.date_range)
            ):
                issues.append(
                    CatalogValidationIssue(
                        code="cross_split_topic_date_overlap",
                        severity="error",
                        message="cross-split families reuse a topic with an overlapping date range",
                        family_ids=pair,
                        similarity=similarity,
                    )
                )

    protected = [prompt.strip() for prompt in protected_prompts if prompt.strip()]
    for entry in entries:
        for protected_prompt in protected:
            similarity = prompt_token_similarity(entry.prompt_text, protected_prompt)
            if _normalized_text(entry.prompt_text) == _normalized_text(protected_prompt) or (
                similarity >= hard_similarity_threshold
            ):
                issues.append(
                    CatalogValidationIssue(
                        code="protected_prompt_leakage",
                        severity="error",
                        message="catalog prompt is too similar to a protected holdout prompt",
                        family_ids=(entry.prompt_family_id,),
                        similarity=similarity,
                    )
                )

    return CatalogValidationReport(
        issues=tuple(issues),
        family_count=len(by_family),
        prompt_count=len(entries),
        split_counts=split_counts,
        max_cross_split_similarity=max_cross_split_similarity,
    )


def _family(
    slug: str,
    period: int,
    skill: ReasoningSkill,
    years: tuple[int, int],
    topic: str,
    prompt: str,
    evidence: tuple[str, str, str],
    misconceptions: tuple[str, str],
    *,
    challenge_2027: bool = False,
) -> PromptFamily:
    family_id = f"p{period}_{slug}"
    return PromptFamily(
        prompt_id=family_id,
        prompt_family_id=family_id,
        prompt_text=prompt,
        source_prompt_text=prompt,
        source_name="APUSH FRQ Grader SLM original prompt bank",
        source_year=2026,
        source_type=PromptSourceType.ORIGINAL,
        period=period,
        reasoning_skill=skill,
        date_range=PromptDateRange(start_year=years[0], end_year=years[1]),
        topic=topic,
        valid_evidence=evidence,
        common_misconceptions=misconceptions,
        format_version=(PromptFormat.MAY_2027_BROAD if challenge_2027 else PromptFormat.STANDARD),
        is_2027_challenge=challenge_2027,
    )


ORIGINAL_PROMPT_FAMILIES: tuple[PromptFamily, ...] = (
    _family(
        "plantation_slavery_growth",
        2,
        ReasoningSkill.CAUSATION,
        (1607, 1754),
        "growth of plantation slavery in British North America",
        "Develop an argument explaining the most important causes of the growth of plantation slavery in British North America from 1607 to 1754.",
        ("Virginia tobacco economy", "Barbados slave codes", "South Carolina rice cultivation"),
        ("Indentured servitude disappeared immediately in 1619", "Slavery developed identically in every colony"),
    ),
    _family(
        "new_england_chesapeake",
        2,
        ReasoningSkill.COMPARISON,
        (1607, 1754),
        "New England and Chesapeake colonial development",
        "Develop an argument comparing the economic and social development of New England and the Chesapeake colonies from 1607 to 1754.",
        ("Puritan family migration", "Chesapeake tobacco exports", "New England town covenants"),
        ("Both regions relied equally on tobacco", "New England had no Atlantic commerce"),
    ),
    _family(
        "indigenous_diplomacy",
        2,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1607, 1754),
        "Indigenous diplomacy with European colonies",
        "Develop an argument evaluating continuity and change in Indigenous diplomatic strategies toward European colonies from 1607 to 1754.",
        ("Powhatan Confederacy", "Haudenosaunee Covenant Chain", "Pueblo Revolt"),
        ("Indigenous nations followed one shared policy", "European settlement ended Native diplomacy by 1620"),
    ),
    _family(
        "religious_pluralism",
        2,
        ReasoningSkill.CAUSATION,
        (1620, 1754),
        "religious pluralism in mainland British colonies",
        "Develop an argument explaining the factors that produced religious pluralism in mainland British colonies from 1620 to 1754.",
        ("Maryland Toleration Act", "Pennsylvania Quaker settlement", "First Great Awakening"),
        ("Every colony established complete religious freedom", "The Great Awakening created colonial churches from nothing"),
    ),
    _family(
        "spanish_british_labor",
        2,
        ReasoningSkill.COMPARISON,
        (1607, 1754),
        "labor systems in Spanish borderlands and British mainland colonies",
        "Develop an argument comparing labor systems in the Spanish borderlands and British mainland colonies from 1607 to 1754.",
        ("Spanish mission labor", "Encomienda legacy", "British indentured servitude"),
        ("Spanish colonies used no coerced labor", "British colonies relied only on free family farms"),
    ),
    _family(
        "colonial_participation",
        2,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1619, 1754),
        "political participation in British mainland colonies",
        "Develop an argument evaluating continuity and change in political participation in British mainland colonies from 1619 to 1754.",
        ("Virginia House of Burgesses", "New England town meetings", "property qualifications for voting"),
        ("Colonial voting included all adults", "Royal governors eliminated elected assemblies"),
    ),
    _family(
        "atlantic_consumer_revolution",
        2,
        ReasoningSkill.CAUSATION,
        (1650, 1754),
        "colonial consumer revolution and Atlantic commerce",
        "Develop an argument explaining how Atlantic commerce caused a consumer revolution in British North America from 1650 to 1754.",
        ("Navigation Acts", "triangular trade", "imported British manufactured goods"),
        ("Colonists became economically independent of Britain", "Atlantic trade involved only Europe and North America"),
    ),
    _family(
        "middle_southern_diversity",
        2,
        ReasoningSkill.COMPARISON,
        (1680, 1754),
        "population diversity in middle and southern colonies",
        "Develop an argument comparing the sources and consequences of population diversity in the middle and southern colonies from 1680 to 1754.",
        ("German migration to Pennsylvania", "Scots-Irish backcountry settlement", "enslaved African majorities in the Lowcountry"),
        ("Southern colonies were culturally uniform", "Middle-colony diversity removed ethnic conflict"),
    ),
    _family(
        "resistance_to_authority",
        3,
        ReasoningSkill.CAUSATION,
        (1754, 1800),
        "colonial and revolutionary resistance to centralized authority",
        "Between 1754 and 1800, disputes over empire, sovereignty, and representation remade political life in British North America and the new United States. Using developments of your choice from this broad period, develop an argument explaining the causes that most transformed Americans' relationship with centralized authority.",
        ("Stamp Act protests", "Declaration of Independence", "Shays' Rebellion"),
        ("Resistance ended once independence was declared", "All colonists opposed centralized power for the same reason"),
        challenge_2027=True,
    ),
    _family(
        "patriot_loyalist_experience",
        3,
        ReasoningSkill.COMPARISON,
        (1775, 1783),
        "Patriot and Loyalist wartime experiences",
        "Develop an argument comparing the political and social experiences of Patriots and Loyalists during the American Revolution from 1775 to 1783.",
        ("state loyalty oaths", "Loyalist property confiscation", "British evacuation of Loyalists"),
        ("Every colonist chose a side", "Loyalists were confined to one social class"),
    ),
    _family(
        "confederation_constitution",
        3,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1776, 1800),
        "national authority from the Articles to the Constitution",
        "Develop an argument evaluating continuity and change in national political authority from 1776 to 1800.",
        ("Articles of Confederation", "Constitutional Convention", "Whiskey Rebellion"),
        ("The Constitution created unlimited federal power", "The Articles had no national institutions"),
    ),
    _family(
        "first_party_system",
        3,
        ReasoningSkill.CAUSATION,
        (1787, 1800),
        "emergence of the first party system",
        "Develop an argument explaining the most important causes of the emergence of the first party system from 1787 to 1800.",
        ("Hamilton's financial program", "debate over the French Revolution", "Alien and Sedition Acts"),
        ("The Constitution formally created political parties", "Federalists and Republicans disagreed on every policy"),
    ),
    _family(
        "revolution_women_enslaved",
        3,
        ReasoningSkill.COMPARISON,
        (1776, 1800),
        "Revolutionary change for women and enslaved African Americans",
        "Develop an argument comparing the effects of Revolutionary ideals on women and enslaved African Americans from 1776 to 1800.",
        ("republican motherhood", "northern gradual emancipation", "petitions for freedom"),
        ("The Revolution granted women suffrage nationwide", "Slavery ended throughout the United States in 1783"),
    ),
    _family(
        "early_republic_diplomacy",
        3,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1776, 1800),
        "United States diplomacy toward European powers",
        "Develop an argument evaluating continuity and change in United States diplomacy toward European powers from 1776 to 1800.",
        ("French alliance of 1778", "Jay Treaty", "Washington's Farewell Address"),
        ("The United States maintained complete neutrality throughout", "The alliance with France continued unchanged after independence"),
    ),
    _family(
        "trans_appalachian_conflict",
        3,
        ReasoningSkill.CAUSATION,
        (1763, 1795),
        "conflict over trans-Appalachian expansion",
        "Develop an argument explaining the causes of conflict over trans-Appalachian expansion from 1763 to 1795.",
        ("Proclamation Line of 1763", "Pontiac's War", "Battle of Fallen Timbers"),
        ("The Proclamation permanently stopped settlement", "Western conflict involved settlers and Britain only"),
    ),
    _family(
        "hamilton_jefferson",
        3,
        ReasoningSkill.COMPARISON,
        (1789, 1800),
        "Hamiltonian and Jeffersonian national visions",
        "Develop an argument comparing Hamiltonian and Jeffersonian visions for the United States from 1789 to 1800.",
        ("Bank of the United States", "Report on Manufactures", "Jeffersonian agrarian republicanism"),
        ("Jefferson opposed all commerce", "Hamilton sought to restore British monarchy"),
    ),
    _family(
        "market_revolution_causes",
        4,
        ReasoningSkill.CAUSATION,
        (1800, 1848),
        "growth of the market revolution",
        "Develop an argument explaining the most important causes of the market revolution from 1800 to 1848.",
        ("Erie Canal", "Lowell textile mills", "steamboat transportation"),
        ("Industrialization began only after the Civil War", "The market revolution reduced regional specialization"),
    ),
    _family(
        "abolition_womens_rights",
        4,
        ReasoningSkill.COMPARISON,
        (1830, 1848),
        "abolitionist and women's rights organizing",
        "Develop an argument comparing the goals and methods of abolitionist and women's rights activists from 1830 to 1848.",
        ("American Anti-Slavery Society", "Grimke sisters", "Seneca Falls Convention"),
        ("The two movements had identical memberships", "Women's rights activism began only after the Civil War"),
    ),
    _family(
        "mass_democracy",
        4,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1800, 1848),
        "mass political participation among white men",
        "Develop an argument evaluating continuity and change in mass political participation among white men from 1800 to 1848.",
        ("removal of property qualifications", "popular presidential campaigning", "second party system"),
        ("Political participation expanded equally for all Americans", "Parties became less organized as suffrage expanded"),
    ),
    _family(
        "antebellum_sectionalism",
        4,
        ReasoningSkill.CAUSATION,
        (1820, 1848),
        "growth of antebellum sectional conflict",
        "Develop an argument explaining the factors that intensified sectional conflict from 1820 to 1848.",
        ("Missouri Compromise", "Nullification Crisis", "Wilmot Proviso"),
        ("Sectional conflict concerned slavery alone", "The Missouri Compromise permanently resolved expansion disputes"),
    ),
    _family(
        "native_removal_resistance",
        4,
        ReasoningSkill.COMPARISON,
        (1800, 1848),
        "Indigenous responses to United States expansion in the Southeast and Old Northwest",
        "Develop an argument comparing Indigenous responses to United States expansion in the Southeast and the Old Northwest from 1800 to 1848.",
        ("Cherokee Nation v. Georgia", "Trail of Tears", "Tecumseh's confederacy"),
        ("Indigenous nations relied only on warfare", "Removal policy affected only the Cherokee"),
    ),
    _family(
        "federal_economic_policy",
        4,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1800, 1848),
        "federal promotion and regulation of economic growth",
        "Develop an argument evaluating continuity and change in federal economic policy from 1800 to 1848.",
        ("Second Bank of the United States", "American System", "Bank War"),
        ("The federal government consistently rejected economic intervention", "Jackson ended all banking in the United States"),
    ),
    _family(
        "evangelical_reform",
        4,
        ReasoningSkill.CAUSATION,
        (1800, 1848),
        "evangelical religion and antebellum reform",
        "Develop an argument explaining how evangelical religion contributed to antebellum reform from 1800 to 1848.",
        ("Second Great Awakening", "temperance societies", "Charles Grandison Finney"),
        ("Revivalism discouraged organized reform", "Every reform movement was exclusively religious"),
    ),
    _family(
        "factory_plantation_labor",
        4,
        ReasoningSkill.COMPARISON,
        (1820, 1848),
        "factory wage labor and plantation enslaved labor",
        "Develop an argument comparing labor organization in northern factories and southern plantations from 1820 to 1848.",
        ("Lowell mills", "task and gang labor systems", "early labor unions"),
        ("Factory workers and enslaved workers had the same legal status", "Plantations produced only cotton"),
    ),
    _family(
        "manifest_destiny",
        5,
        ReasoningSkill.CAUSATION,
        (1844, 1860),
        "territorial expansion in the 1840s and 1850s",
        "Develop an argument explaining the causes of United States territorial expansion from 1844 to 1860.",
        ("annexation of Texas", "Oregon boundary settlement", "Mexican-American War"),
        ("Expansion was motivated only by land hunger", "All acquired territory immediately became free states"),
    ),
    _family(
        "union_confederate_mobilization",
        5,
        ReasoningSkill.COMPARISON,
        (1861, 1865),
        "Union and Confederate wartime mobilization",
        "Develop an argument comparing Union and Confederate strategies for mobilizing people and resources from 1861 to 1865.",
        ("Union conscription", "Confederate impressment", "Legal Tender Act"),
        ("Neither government used conscription", "The Confederacy possessed greater industrial capacity"),
    ),
    _family(
        "black_freedom_reconstruction",
        5,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1863, 1877),
        "African American freedom during emancipation and Reconstruction",
        "Develop an argument evaluating continuity and change in African American freedom from 1863 to 1877.",
        ("Thirteenth Amendment", "Freedmen's Bureau", "Black Codes"),
        ("Emancipation immediately produced economic equality", "Reconstruction eliminated racial violence"),
    ),
    _family(
        "civil_war_causes",
        5,
        ReasoningSkill.CAUSATION,
        (1844, 1861),
        "coming of the Civil War",
        "Develop an argument explaining the factors that most directly caused the Civil War from 1844 to 1861.",
        ("Compromise of 1850", "Kansas-Nebraska Act", "election of 1860"),
        ("The war began primarily over tariffs", "Sectional parties existed unchanged since 1789"),
    ),
    _family(
        "reconstruction_plans",
        5,
        ReasoningSkill.COMPARISON,
        (1865, 1877),
        "presidential and congressional Reconstruction",
        "Develop an argument comparing presidential and congressional approaches to Reconstruction from 1865 to 1877.",
        ("Andrew Johnson's restoration policy", "Reconstruction Acts", "Fourteenth Amendment"),
        ("Lincoln and Johnson proposed identical plans", "Congress controlled Reconstruction from the moment the war ended"),
    ),
    _family(
        "federal_state_power",
        5,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1844, 1877),
        "federal and state power during sectional crisis and Reconstruction",
        "Develop an argument evaluating continuity and change in the balance of federal and state power from 1844 to 1877.",
        ("Fugitive Slave Act", "Civil War suspension of habeas corpus", "Reconstruction Acts"),
        ("States' rights arguments belonged only to the South", "Federal authority expanded at a constant rate"),
    ),
    _family(
        "reconstruction_collapse",
        5,
        ReasoningSkill.CAUSATION,
        (1868, 1877),
        "decline of Reconstruction governments",
        "Develop an argument explaining the causes of the decline of Reconstruction governments from 1868 to 1877.",
        ("Ku Klux Klan violence", "Panic of 1873", "Compromise of 1877"),
        ("Reconstruction ended because its goals were fully achieved", "One presidential election alone caused its decline"),
    ),
    _family(
        "western_communities",
        5,
        ReasoningSkill.COMPARISON,
        (1848, 1877),
        "United States expansion and Indigenous and Mexican American communities",
        "Develop an argument comparing the effects of United States western expansion on Indigenous and Mexican American communities from 1848 to 1877.",
        ("Treaty of Guadalupe Hidalgo", "reservation policy", "California Land Act of 1851"),
        ("Both groups held identical treaty rights", "Western expansion displaced only Indigenous peoples"),
    ),
    _family(
        "industrial_consolidation",
        6,
        ReasoningSkill.CAUSATION,
        (1865, 1898),
        "industrial consolidation and large corporations",
        "Develop an argument explaining the causes of industrial consolidation in the United States from 1865 to 1898.",
        ("transcontinental railroads", "vertical integration", "protective tariffs"),
        ("Large corporations grew without government assistance", "Industrial consolidation eliminated competition everywhere"),
    ),
    _family(
        "knights_afl",
        6,
        ReasoningSkill.COMPARISON,
        (1869, 1898),
        "Knights of Labor and American Federation of Labor",
        "Develop an argument comparing the membership strategies and goals of the Knights of Labor and the American Federation of Labor from 1869 to 1898.",
        ("inclusive Knights membership", "Haymarket affair", "AFL craft unionism"),
        ("Both organizations accepted the same workers", "The AFL rejected collective bargaining"),
    ),
    _family(
        "urban_immigrant_communities",
        6,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1865, 1898),
        "immigrant community life in industrial cities",
        "Develop an argument evaluating continuity and change in immigrant community life in industrial cities from 1865 to 1898.",
        ("ethnic enclaves", "political machines", "settlement houses"),
        ("New immigrants abandoned their cultures immediately", "Urban political machines offered no social services"),
    ),
    _family(
        "agrarian_protest",
        6,
        ReasoningSkill.CAUSATION,
        (1865, 1896),
        "rise of organized agrarian protest",
        "Develop an argument explaining the causes of organized agrarian protest from 1865 to 1896.",
        ("railroad freight rates", "crop-lien system", "free silver movement"),
        ("Farmers opposed every form of federal regulation", "Agrarian protest was limited to the Northeast"),
    ),
    _family(
        "south_west_racial_order",
        6,
        ReasoningSkill.COMPARISON,
        (1877, 1898),
        "racial control in the New South and the trans-Mississippi West",
        "Develop an argument comparing systems of racial control in the New South and the trans-Mississippi West from 1877 to 1898.",
        ("Jim Crow laws", "Dawes Act", "Chinese Exclusion Act"),
        ("Racial policy targeted only African Americans", "Western federal policy consistently protected tribal sovereignty"),
    ),
    _family(
        "gilded_age_regulation",
        6,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1865, 1898),
        "federal regulation of the national economy",
        "Develop an argument evaluating continuity and change in federal regulation of the national economy from 1865 to 1898.",
        ("Interstate Commerce Act", "Sherman Antitrust Act", "Supreme Court limits on regulation"),
        ("The federal government never regulated business before 1900", "Antitrust law immediately broke up every trust"),
    ),
    _family(
        "overseas_expansion",
        6,
        ReasoningSkill.CAUSATION,
        (1880, 1898),
        "United States overseas expansion",
        "Develop an argument explaining the causes of United States overseas expansion from 1880 to 1898.",
        ("navalism associated with Alfred Thayer Mahan", "Hawaiian annexation movement", "Spanish-American War"),
        ("Overseas expansion began only after 1898", "Economic motives were the sole cause"),
    ),
    _family(
        "progressive_reform",
        7,
        ReasoningSkill.CAUSATION,
        (1890, 1920),
        "rise of Progressive reform",
        "Develop an argument explaining the causes of Progressive reform from 1890 to 1920.",
        ("muckraking journalism", "Social Gospel", "urban political corruption"),
        ("Progressives shared one unified program", "Progressive reform focused only on rural problems"),
    ),
    _family(
        "depression_federal_approaches",
        7,
        ReasoningSkill.COMPARISON,
        (1929, 1941),
        "federal responses to the Great Depression",
        "From 1890 to 1945, Americans confronted repeated debates over whether national power should protect economic security. Selecting relevant policies from the Great Depression era, develop an argument comparing two distinct federal approaches to economic crisis.",
        ("Reconstruction Finance Corporation", "Federal Emergency Relief Administration", "Social Security Act"),
        ("Hoover refused every federal response", "New Deal programs ended the Depression by themselves"),
        challenge_2027=True,
    ),
    _family(
        "wartime_civil_liberties",
        7,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1917, 1945),
        "civil liberties during the two world wars",
        "Develop an argument evaluating continuity and change in federal restrictions on civil liberties during wartime from 1917 to 1945.",
        ("Espionage and Sedition Acts", "Schenck v. United States", "Japanese American incarceration"),
        ("The Bill of Rights prevented all wartime restrictions", "The same groups faced identical policies in both wars"),
    ),
    _family(
        "world_war_one_entry",
        7,
        ReasoningSkill.CAUSATION,
        (1914, 1917),
        "United States entry into the First World War",
        "Develop an argument explaining the causes of United States entry into the First World War from 1914 to 1917.",
        ("unrestricted submarine warfare", "Zimmermann Telegram", "American loans to the Allies"),
        ("The Lusitania immediately caused a declaration of war", "The United States was economically neutral"),
    ),
    _family(
        "great_dust_migrations",
        7,
        ReasoningSkill.COMPARISON,
        (1915, 1940),
        "Great Migration and Dust Bowl migration",
        "Develop an argument comparing the causes and consequences of the Great Migration and Dust Bowl migration from 1915 to 1940.",
        ("wartime industrial jobs", "Jim Crow violence", "Okie migration to California"),
        ("Both migrations originated in the Great Plains", "Migrants encountered no discrimination in destination regions"),
    ),
    _family(
        "social_welfare_state",
        7,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1890, 1945),
        "federal responsibility for social welfare",
        "Develop an argument evaluating continuity and change in federal responsibility for social welfare from 1890 to 1945.",
        ("Freedmen's pension proposals", "Progressive labor regulation", "Social Security Act"),
        ("Federal welfare policy began suddenly in 1933", "New Deal benefits reached all groups equally"),
    ),
    _family(
        "world_war_two_homefront",
        7,
        ReasoningSkill.CAUSATION,
        (1941, 1945),
        "social and economic changes on the Second World War home front",
        "Develop an argument explaining how wartime mobilization caused social and economic change in the United States from 1941 to 1945.",
        ("War Production Board", "Rosie the Riveter", "Double V campaign"),
        ("Mobilization ended racial discrimination", "Women permanently kept all wartime industrial jobs"),
    ),
    _family(
        "containment_origins",
        8,
        ReasoningSkill.CAUSATION,
        (1945, 1960),
        "development of United States containment policy",
        "Develop an argument explaining the causes of United States containment policy from 1945 to 1960.",
        ("Truman Doctrine", "Marshall Plan", "formation of NATO"),
        ("Containment required direct war with the Soviet Union", "United States leaders agreed on every Cold War policy"),
    ),
    _family(
        "civil_rights_strategies",
        8,
        ReasoningSkill.COMPARISON,
        (1954, 1968),
        "legal and direct-action civil rights strategies",
        "Develop an argument comparing legal challenges and mass direct action in the Black freedom struggle from 1954 to 1968.",
        ("Brown v. Board of Education", "Montgomery bus boycott", "Birmingham campaign"),
        ("Court victories automatically enforced integration", "Direct-action campaigns rejected all legal work"),
    ),
    _family(
        "federal_civil_rights_role",
        8,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1945, 1980),
        "federal enforcement of civil rights",
        "Develop an argument evaluating continuity and change in the federal government's role in enforcing civil rights from 1945 to 1980.",
        ("Executive Order 9981", "Civil Rights Act of 1964", "Regents of the University of California v. Bakke"),
        ("Federal enforcement advanced steadily without resistance", "Civil rights policy concerned voting alone"),
    ),
    _family(
        "conservative_resurgence",
        8,
        ReasoningSkill.CAUSATION,
        (1964, 1980),
        "rise of the modern conservative movement",
        "Develop an argument explaining the causes of the modern conservative movement's growth from 1964 to 1980.",
        ("Barry Goldwater campaign", "tax revolt", "Moral Majority"),
        ("Conservatism emerged only in 1980", "All conservatives opposed every New Deal program"),
    ),
    _family(
        "korea_vietnam_homefront",
        8,
        ReasoningSkill.COMPARISON,
        (1950, 1975),
        "domestic effects of the Korean and Vietnam Wars",
        "Develop an argument comparing the domestic political effects of the Korean and Vietnam Wars from 1950 to 1975.",
        ("Truman-MacArthur controversy", "Gulf of Tonkin Resolution", "War Powers Resolution"),
        ("Both wars produced equally large antiwar movements", "Congress played no role in either conflict"),
    ),
    _family(
        "post_1965_immigration",
        8,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1965, 1980),
        "immigration patterns after the 1965 immigration law",
        "Develop an argument evaluating continuity and change in United States immigration patterns from 1965 to 1980.",
        ("Immigration and Nationality Act of 1965", "family reunification preferences", "increased Asian immigration"),
        ("The 1965 law created open borders", "European immigration ended completely"),
    ),
    _family(
        "environmental_movement",
        8,
        ReasoningSkill.CAUSATION,
        (1945, 1980),
        "growth of the modern environmental movement",
        "Develop an argument explaining the causes of the modern environmental movement from 1945 to 1980.",
        ("Silent Spring", "Santa Barbara oil spill", "first Earth Day"),
        ("Environmental activism began with Earth Day", "The movement opposed all federal regulation"),
    ),
    _family(
        "deindustrialization",
        9,
        ReasoningSkill.CAUSATION,
        (1980, 2000),
        "deindustrialization in older manufacturing regions",
        "Develop an argument explaining the causes of deindustrialization in older United States manufacturing regions from 1980 to 2000.",
        ("automation", "global manufacturing competition", "decline of unionized steel employment"),
        ("Trade policy alone caused every factory closure", "Manufacturing output disappeared from the United States"),
    ),
    _family(
        "reagan_bush_economics",
        9,
        ReasoningSkill.COMPARISON,
        (1981, 2008),
        "Reagan and George W. Bush economic policies",
        "Develop an argument comparing the economic policies of the Reagan and George W. Bush administrations from 1981 to 2008.",
        ("Economic Recovery Tax Act", "deregulation", "2001 and 2003 tax cuts"),
        ("Both administrations balanced the federal budget", "Their economic programs eliminated federal regulation"),
    ),
    _family(
        "immigration_politics",
        9,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1980, 2020),
        "immigration policy and political debate",
        "Since 1980, migration, labor demand, border enforcement, and ideas about national identity have repeatedly shaped public debate. Using developments of your choice from this broad era, develop an argument evaluating continuity and change in United States immigration policy and politics from 1980 to 2020.",
        ("Immigration Reform and Control Act of 1986", "Deferred Action for Childhood Arrivals", "post-2001 border enforcement"),
        ("The 1986 law only increased enforcement", "Immigration politics followed a single party divide throughout the period"),
        challenge_2027=True,
    ),
    _family(
        "cold_war_end",
        9,
        ReasoningSkill.CAUSATION,
        (1980, 1991),
        "end of the Cold War",
        "Develop an argument explaining the factors that contributed to the end of the Cold War from 1980 to 1991.",
        ("Reagan-era military buildup", "Gorbachev's reforms", "fall of the Berlin Wall"),
        ("One United States policy alone ended the Cold War", "The Soviet Union dissolved immediately after Reagan's election"),
    ),
    _family(
        "identity_movements",
        9,
        ReasoningSkill.COMPARISON,
        (1980, 2015),
        "LGBTQ rights and immigrant-rights movements",
        "Develop an argument comparing the strategies of LGBTQ rights and immigrant-rights movements from 1980 to 2015.",
        ("ACT UP", "marriage-equality litigation", "2006 immigrant-rights marches"),
        ("Both movements relied only on court cases", "Neither movement built national coalitions"),
    ),
    _family(
        "post_cold_war_conflict",
        9,
        ReasoningSkill.CONTINUITY_CHANGE,
        (1980, 2011),
        "United States military involvement abroad",
        "Develop an argument evaluating continuity and change in United States military involvement abroad from 1980 to 2011.",
        ("Persian Gulf War", "Kosovo intervention", "Authorization for Use of Military Force of 2001"),
        ("The Cold War's end ended overseas intervention", "Every post-1980 conflict involved a formal declaration of war"),
    ),
    _family(
        "political_polarization",
        9,
        ReasoningSkill.CAUSATION,
        (1980, 2020),
        "growth of national political polarization",
        "Develop an argument explaining the causes of growing national political polarization from 1980 to 2020.",
        ("rise of partisan cable media", "1994 Republican congressional victory", "geographic party sorting"),
        ("Social media created polarization by itself", "Party coalitions remained unchanged after 1980"),
    ),
)


def build_default_prompt_catalog(
    *, seed: str = DEFAULT_SPLIT_SEED
) -> list[PromptCatalogEntry]:
    """Build and validate the version-one project-authored prompt catalog."""
    entries = assign_family_splits(ORIGINAL_PROMPT_FAMILIES, seed=seed)
    validate_prompt_catalog(entries).raise_for_issues()
    return entries
