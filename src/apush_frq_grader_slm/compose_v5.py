"""Score-blind timed-student LEQ essay composer for v5.

Generates authentic student essays from blinded writer packets without seeing
score targets or rubric points. Capability / composition profiles and optional
``observable_writing_behavior`` (boundary tasks) drive quality and structure.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from apush_frq_grader_slm.compose_v4 import rng_for_task
from apush_frq_grader_slm.fact_cards_v5 import has_eight_gram_overlap
from apush_frq_grader_slm.ingest.dedup import normalize_essay

GENERATOR_NAME = "compose_v5_score_blind"

_SCORE_LEAK_KEYS = frozenset(
    {
        "target_scores",
        "target_total",
        "scores",
        "score",
        "rubric_text",
        "resolved_grade",
        "reference_scores",
    }
)

_MISSPELL_DRAFT: tuple[tuple[str, str], ...] = (
    ("because", "becuase"),
    ("government", "goverment"),
    ("which", "wich"),
    ("their", "thier"),
    ("necessary", "neccessary"),
    ("separate", "seperate"),
    ("argument", "arguement"),
    ("throughout", "throughtout"),
    ("significant", "signifigant"),
    ("therefore", "therefor"),
    ("different", "diffrent"),
    ("especially", "expecially"),
)

_PARAPHRASE_SWAPS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\brenegotiated the papal demarcation and formalized it as\b", re.I), "shifted the dividing line and called it"),
    (re.compile(r"\bformalized it as\b", re.I), "ended up calling it"),
    (re.compile(r"\bcodified a settlement known as\b", re.I), "made a deal people call"),
    (re.compile(r"\benforced mercantile controls through\b", re.I), "tried to control trade with"),
    (re.compile(r"\bbuilt an expansive imperial order\b", re.I), "built up a bigger empire"),
    (re.compile(r"\bdemographic growth proved still sharper\b", re.I), "population grew even faster"),
    (re.compile(r"\bimproved waterways lowered freight costs\b", re.I), "better canals and rivers made shipping cheaper"),
    (re.compile(r"\bcontributed to\b", re.I), "helped cause"),
    (re.compile(r"\bled to\b", re.I), "ended up causing"),
    (re.compile(r"\bresulted in\b", re.I), "meant that"),
    (re.compile(r"\bsignificantly\b", re.I), "a lot"),
    (re.compile(r"\bsubstantially\b", re.I), "pretty clearly"),
    (re.compile(r"\bfacilitated\b", re.I), "made easier"),
    (re.compile(r"\bestablished\b", re.I), "set up"),
    (re.compile(r"\bimplemented\b", re.I), "put in place"),
    (re.compile(r"\btransformed\b", re.I), "changed"),
    (re.compile(r"\breoriented\b", re.I), "shifted"),
    (re.compile(r"\bexpansion of\b", re.I), "growth of"),
    (re.compile(r"\bdevelopment of\b", re.I), "rise of"),
    (re.compile(r"\bincreasingly\b", re.I), "more and more"),
    (re.compile(r"\bconsequently\b", re.I), "so"),
    (re.compile(r"\bfurthermore\b", re.I), "also"),
    (re.compile(r"\bmoreover\b", re.I), "plus"),
    (re.compile(r"\bhowever\b", re.I), "but"),
    (re.compile(r"\balthough\b", re.I), "even though"),
)

_GENERIC_EVIDENCE = (
    "new laws and court fights",
    "local protests and petitions",
    "shifts in trade and work",
    "debates in Congress",
    "migration and settlement pressure",
    "wartime mobilization",
    "party politics and elections",
    "reform campaigns",
)

# Leading tokens that look like Proper names but are not real entities.
_BAD_ENTITY_TOKENS = frozenset(
    {
        "the",
        "a",
        "an",
        "by",
        "in",
        "on",
        "at",
        "of",
        "to",
        "for",
        "from",
        "with",
        "as",
        "and",
        "or",
        "after",
        "before",
        "during",
        "under",
        "over",
        "into",
        "about",
        "one",
        "two",
        "most",
        "some",
        "both",
        "this",
        "that",
        "these",
        "those",
        "source",
        "organized",
        "secretary",
        "period",
        "originally",
        "another",
        "also",
        "when",
        "while",
        "where",
        "what",
        "which",
        "who",
        "their",
        "there",
        "his",
        "her",
        "its",
        "our",
        "your",
        "is",
        "was",
        "were",
        "are",
        "been",
        "being",
    }
)

_TOPIC_VERBS = re.compile(
    r"\b(?:molded|shaped|remade|changed|transformed|affected|influenced|"
    r"impacted|altered|reoriented|drove|caused|created|produced|rebuilt|"
    r"redefined|structured|organized|challenged|reinforced|weakened|"
    r"strengthened|expanded|limited|restricted|encouraged|"
    r"contributed to|led to|resulted in|helped cause)\b",
    re.I,
)

_YEAR_RANGE_RE = re.compile(
    r"\b(?:from|between)\s+(1[4-9]\d{2}|20[0-2]\d)\s+(?:to|and|-|–|—)\s+"
    r"(1[4-9]\d{2}|20[0-2]\d)\b",
    re.I,
)

_YEAR_SPAN_RE = re.compile(
    r"\b(1[4-9]\d{2}|20[0-2]\d)\s*[-–—]\s*(1[4-9]\d{2}|20[0-2]\d)\b"
)

_MEMORY_PREFIX_RE = re.compile(
    r"^(i remember(?: something like)?|i kind of remember|class notes had|"
    r"notes mentioned|there was also|people talk about how|"
    r"one thing that comes up is)\s+",
    re.I,
)

_REF_POINT_RE = re.compile(
    r"^(?P<sub>.+?)\s+is a remembered reference point"
    r"(?:\s+in period\s+\d+)?(?:\s*\((?P<year>\d{4})\))?\.?$",
    re.I,
)


def compose_essay(
    packet: Mapping[str, Any],
    *,
    observable_writing_behavior: str | None = None,
    rng: Any | None = None,
) -> str:
    """Compose one timed-student essay from a score-blind writer packet.

    Parameters
    ----------
    packet:
        Writer packet with ``prompt``, ``student_capability``,
        ``timed_composition_style``, ``style_reference``, and
        ``semantic_fact_cards``. Must not rely on score targets.
    observable_writing_behavior:
        Optional boundary cue restored from the private task plan when absent
        from the packet. If omitted, uses
        ``packet['student_capability']['observable_writing_behavior']`` when set.
    rng:
        Deterministic ``random.Random``. Defaults to ``rng_for_task(task_id)``.
    """
    _assert_score_blind(packet)
    task_id = str(packet.get("task_id") or "v5-anon")
    if rng is None:
        rng = rng_for_task(task_id)

    prompt = str(packet.get("prompt") or "").strip()
    capability = dict(packet.get("student_capability") or {})
    composition = dict(packet.get("timed_composition_style") or {})
    style_ref = str(packet.get("style_reference") or "")[:400]
    cards = list(packet.get("semantic_fact_cards") or [])

    behavior = (
        observable_writing_behavior
        or str(capability.get("observable_writing_behavior") or "").strip()
        or None
    )

    knowledge = str(capability.get("historical_knowledge") or "competent")
    argument = str(capability.get("argument_control") or "partial")
    time_pressure = str(composition.get("time_pressure") or "normal")
    mechanics = str(composition.get("mechanics") or "minor_errors")
    organization = str(composition.get("organization") or "clear")

    topic = _topic_phrase(prompt)
    remembered = _remembered_concepts(cards, knowledge, rng)
    cues = _style_cues(style_ref)

    lo, hi = _length_band(knowledge, time_pressure)
    intents = _behavior_intents(behavior, knowledge, argument)

    paragraphs = _draft_paragraphs(
        prompt=prompt,
        topic=topic,
        remembered=remembered,
        knowledge=knowledge,
        argument=argument,
        organization=organization,
        time_pressure=time_pressure,
        mechanics=mechanics,
        intents=intents,
        cues=cues,
        rng=rng,
    )

    essay = _join_paragraphs(paragraphs, organization, rng)
    essay = _bake_mechanics_inplace(essay, mechanics, organization, rng)
    essay = _fit_length(essay, lo, hi, topic, remembered, knowledge, rng)
    essay = _scrub_eight_gram_copies(essay, cards, style_ref, rng)
    essay = _cleanup_essay(essay, knowledge=knowledge, topic=topic, remembered=remembered, rng=rng)
    return essay.strip()


def resolve_observable_behavior(
    packet: Mapping[str, Any],
    task: Mapping[str, Any] | None = None,
) -> str | None:
    """Prefer packet capability cue; else restore from private task metadata."""
    capability = dict(packet.get("student_capability") or {})
    from_packet = str(capability.get("observable_writing_behavior") or "").strip()
    if from_packet:
        return from_packet
    if not task:
        return None
    task_cap = dict(task.get("capability_profile") or {})
    from_task = str(task_cap.get("observable_writing_behavior") or "").strip()
    return from_task or None


def _assert_score_blind(packet: Mapping[str, Any]) -> None:
    leaked = _SCORE_LEAK_KEYS & set(packet)
    if leaked:
        raise ValueError(f"writer packet leaked scoring keys: {sorted(leaked)}")


def _topic_phrase(prompt: str) -> str:
    """Return a short topic noun phrase (about 8–12 words), not a near-full prompt."""
    raw = prompt.strip().rstrip(".!?")
    years = ""
    m_range = _YEAR_RANGE_RE.search(raw)
    if m_range:
        years = f"{m_range.group(1)}–{m_range.group(2)}"
        raw = (raw[: m_range.start()] + " " + raw[m_range.end() :]).strip()
    else:
        m_span = _YEAR_SPAN_RE.search(raw)
        if m_span:
            years = f"{m_span.group(1)}–{m_span.group(2)}"
            raw = (raw[: m_span.start()] + " " + raw[m_span.end() :]).strip()

    text = re.sub(
        r"^(Evaluate|Analyze|Compare|Explain|Assess|Examine|Determine|Judge|"
        r"Weigh|Rank|Trace|Discuss)\s+"
        r"(?:the extent to which|how far|how|why|whether|the degree to which|"
        r"the ways|the leading factors behind|the main causes of|the force of|"
        r"whether|if)\s+",
        "",
        raw,
        flags=re.I,
    )
    text = re.sub(
        r"^(Evaluate|Analyze|Compare|Explain|Assess|Examine|Determine|Judge|"
        r"Weigh|Rank|Trace|Discuss)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip(" ,;:")

    focus = _compress_topic_focus(text)
    if years:
        phrase = f"{focus}, {years}" if focus else years
    else:
        phrase = focus or "this historical development"

    words = phrase.split()
    if len(words) > 12:
        if years and phrase.endswith(years):
            head = " ".join(words[: max(1, 12 - len(years.split()))])
            phrase = f"{head.rstrip(',;')}, {years}"
        else:
            phrase = " ".join(words[:12])
    phrase = phrase.strip(" ,;:")
    if not phrase:
        return "this historical development"
    return phrase[0].lower() + phrase[1:]


def _compress_topic_focus(text: str) -> str:
    """Turn a stripped prompt clause into a short noun-ish focus."""
    text = text.strip()
    if not text:
        return ""

    parts = _TOPIC_VERBS.split(text, maxsplit=1)
    if len(parts) == 2:
        left = _trim_topic_np(parts[0].strip(" ,;:"), max_words=5)
        right = parts[1].strip(" ,;:")
        right = re.sub(
            r"^(?:the\s+)?(?:United States|American|British North American)\s+",
            "",
            right,
            flags=re.I,
        )
        right = _trim_topic_np(right, max_words=4)
        if left and right:
            joined = f"{left} and {right}"
        else:
            joined = left or right
        return _trim_topic_np(joined, max_words=10)

    m = re.match(
        r"^(?:the\s+)?(?:expansion|growth|rise|role|impact|influence|force|"
        r"development|effects?|consequences?)\s+of\s+(.+)$",
        text,
        flags=re.I,
    )
    if m:
        return _trim_topic_np(m.group(1), max_words=8)

    return _trim_topic_np(text, max_words=10)


def _trim_topic_np(text: str, *, max_words: int) -> str:
    text = re.sub(r"\s+", " ", text).strip(" ,;:.")
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I)
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words).strip(" ,;:")


def _topic_slot(topic: str, *, article: str | None = None) -> str:
    """Insert topic without producing 'the the' / 'a a'."""
    t = topic.strip()
    if not article:
        return t
    bare = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.I)
    return f"{article} {bare}".strip()


def _length_band(knowledge: str, time_pressure: str) -> tuple[int, int]:
    bands = {
        "limited": (120, 155),
        "uneven": (145, 195),
        "competent": (175, 235),
        "strong": (210, 280),
    }
    lo, hi = bands.get(knowledge, (150, 210))
    if time_pressure == "severe":
        hi = min(hi, lo + 35)
    elif time_pressure == "moderate":
        hi = min(hi, lo + 55)
    return lo, hi


def _remembered_concepts(
    cards: Sequence[Mapping[str, Any]],
    knowledge: str,
    rng: Any,
) -> list[str]:
    concepts = []
    for card in cards:
        concept = str(card.get("concept") or card.get("fact") or "").strip()
        if concept:
            paraphrased = _paraphrase_concept(concept, rng)
            if paraphrased:
                concepts.append(paraphrased)
    rng.shuffle(concepts)
    keep = {"limited": 1, "uneven": 2, "competent": 3, "strong": 5}.get(knowledge, 2)
    if knowledge == "limited":
        concepts = concepts[:1]
        if not concepts:
            concepts = [rng.choice(_GENERIC_EVIDENCE)]
        return concepts
    if not concepts:
        pool = list(_GENERIC_EVIDENCE)
        rng.shuffle(pool)
        return pool[:keep]
    return concepts[:keep]


def _paraphrase_concept(concept: str, rng: Any) -> str:
    """Student-memory paraphrase that keeps real entities and never ends in 'mattering'."""
    text = concept.strip().rstrip(".")
    if not text:
        return ""

    ref = _REF_POINT_RE.match(text)
    if ref:
        entity = _clean_entity_name(ref.group("sub") or "")
        year = ref.group("year")
        if entity:
            return _memory_clause_for_entity(entity, year, rng)
        if year:
            return rng.choice(
                (
                    f"i remember something from around {year} coming up in notes",
                    f"class notes mentioned developments around {year}",
                    f"notes mentioned pressure building around {year}",
                )
            )
        return ""

    for pattern, repl in _PARAPHRASE_SWAPS:
        text = pattern.sub(repl, text)

    body = text[0].lower() + text[1:] if text else text
    body = body.rstrip(" .")
    words = body.split()
    if len(words) > 18:
        body = " ".join(words[:18])

    if _MEMORY_PREFIX_RE.match(body):
        paraphrased = body
    else:
        wrappers = (
            "i remember something like {c}",
            "class notes had {c}",
            "there was also {c}",
            "people talk about how {c}",
            "one thing that comes up is {c}",
            "notes mentioned {c}",
        )
        paraphrased = rng.choice(wrappers).format(c=body)

    if has_eight_gram_overlap(paraphrased, concept):
        entities = _extract_entity_names(concept)
        years = re.findall(r"\b(?:1[4-9]\d{2}|20[0-2]\d)\b", concept)
        year = years[0] if years else None
        if entities:
            paraphrased = _memory_clause_for_entity(entities[0], year, rng)
            if len(entities) > 1 and rng.random() < 0.45:
                paraphrased = (
                    f"notes mentioned {entities[0]} and {entities[1]}"
                    + (f" around {year}" if year else "")
                )
        elif year:
            paraphrased = rng.choice(
                (
                    f"i remember something from around {year} changing the period",
                    f"class notes mentioned developments around {year}",
                )
            )
        else:
            paraphrased = rng.choice(_GENERIC_EVIDENCE)

    paraphrased = re.sub(r"\bmattering\b", "", paraphrased, flags=re.I)
    paraphrased = re.sub(r"\s{2,}", " ", paraphrased).strip(" ,;.")
    return paraphrased


def _clean_entity_name(name: str) -> str:
    """Strip leading articles/prepositions; require a real-looking entity."""
    parts = re.findall(r"[A-Za-z0-9'’\-]+", name or "")
    while parts and parts[0].lower() in _BAD_ENTITY_TOKENS:
        parts = parts[1:]
    while parts and parts[-1].lower() in _BAD_ENTITY_TOKENS:
        parts = parts[:-1]
    if not parts:
        return ""
    if len(parts[0]) <= 2 or parts[0].lower() in _BAD_ENTITY_TOKENS:
        return ""
    return " ".join(parts)


def _extract_entity_names(concept: str) -> list[str]:
    """Pull Proper-looking spans that survive entity hygiene."""
    found: list[str] = []
    for match in re.finditer(
        r"\b([A-Z][A-Za-z0-9'’\-]+(?:\s+[A-Z][A-Za-z0-9'’\-]+){0,4})\b",
        concept,
    ):
        cleaned = _clean_entity_name(match.group(1))
        if cleaned and cleaned not in found:
            found.append(cleaned)
    return found


def _memory_clause_for_entity(entity: str, year: str | None, rng: Any) -> str:
    """Full student clause — never 'X mattering' or 'around YEAR mattering'."""
    needs_the = bool(
        re.match(
            r"^(Treaty|War|Election|Revolution|Constitution|Compromise|"
            r"Purchase|Doctrine|Act|Bank|System|Movement|Party)\b",
            entity,
            flags=re.I,
        )
        or re.search(r"\b(War|Treaty|Revolution|Compromise|Purchase)\b", entity)
    )
    label = f"the {entity}" if needs_the and not entity.lower().startswith("the ") else entity
    if year:
        options = (
            f"i remember {label} in {year} changing foreign policy",
            f"i remember {label} around {year} coming up in class",
            f"notes mentioned {label} and pressures around {year}",
            f"class notes had {label} tied to {year}",
            f"people talk about how {label} in {year} shifted politics",
        )
    else:
        options = (
            f"i remember {label} changing the period",
            f"notes mentioned {label} and postwar loyalty fears",
            f"class notes had {label} as a key example",
            f"people talk about how {label} shaped events",
            f"one thing that comes up is {label}",
        )
    return rng.choice(options)


def _style_cues(style_ref: str) -> dict[str, Any]:
    """Extract light style cues without copying long verbatim spans."""
    text = style_ref.strip()
    cues: dict[str, Any] = {
        "informal": bool(re.search(r"\b(i think|kinda|wasnt|dont|im)\b", text, re.I)),
        "starts_lower": bool(text) and text[0].islower(),
        "uses_although": bool(re.search(r"\balthough\b", text, re.I)),
        "short_burst": False,
    }
    # Capture at most one short distinctive 2–3 word cue, never a long span.
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) >= 3:
        # Prefer mid-excerpt rhythm words rather than opening thesis clone.
        mid = max(0, len(words) // 3)
        cue_words = words[mid : mid + 2]
        if cue_words and all(len(w) < 12 for w in cue_words):
            cues["micro_phrase"] = " ".join(cue_words).lower()
    return cues


def _behavior_intents(
    behavior: str | None,
    knowledge: str,
    argument: str,
) -> dict[str, str]:
    """Map observable behavior (or capability defaults) into drafting intents."""
    intents = {
        "thesis": "weak" if argument in {"emerging", "partial"} else "clear",
        "context": "skip" if knowledge == "limited" else "light",
        "evidence": "vague" if knowledge == "limited" else ("list" if knowledge == "uneven" else "linked"),
        "analysis": "flat" if argument in {"emerging", "partial"} else "causal",
        "complexity": "none",
    }
    if not behavior:
        if argument == "nuanced":
            intents["complexity"] = "qualify"
        return intents

    b = behavior.lower()
    if "never settles on one overall answer" in b:
        intents["thesis"] = "unsettle"
    elif "states one overall answer" in b:
        intents["thesis"] = "clear"
    if "starts directly with the period" in b:
        intents["context"] = "skip"
    elif "earlier or broader development" in b:
        intents["context"] = "prior"
    if "at most one concrete" in b or "broad trends" in b:
        intents["evidence"] = "one_or_vague"
    elif "at least two concrete" in b and "unevenly" in b:
        intents["evidence"] = "two_uneven"
    elif "without consistently using them to prove" in b:
        intents["evidence"] = "names_only"
    elif "explains how concrete examples support" in b:
        intents["evidence"] = "linked"
    if "without organizing them through a historical relationship" in b:
        intents["analysis"] = "list"
    elif "organizes the explanation around cause, comparison, or change" in b:
        intents["analysis"] = "causal"
    elif "stays straightforward" in b:
        intents["analysis"] = "causal"
        intents["complexity"] = "none"
    elif "qualifies the argument" in b or "multiple connected perspectives" in b:
        intents["analysis"] = "causal"
        intents["complexity"] = "qualify"
    return intents


def _draft_paragraphs(
    *,
    prompt: str,
    topic: str,
    remembered: list[str],
    knowledge: str,
    argument: str,
    organization: str,
    time_pressure: str,
    mechanics: str,
    intents: dict[str, str],
    cues: dict[str, Any],
    rng: Any,
) -> list[str]:
    paras: list[str] = []

    if intents["context"] == "prior":
        paras.append(_context_prior(topic, remembered, mechanics, rng))
    elif intents["context"] == "light" and knowledge in {"competent", "strong"}:
        if rng.random() < 0.65:
            paras.append(_context_prior(topic, remembered, mechanics, rng))

    paras.append(_thesis_paragraph(prompt, topic, remembered, intents["thesis"], cues, mechanics, rng))

    evid = remembered[:]
    rng.shuffle(evid)
    body = _evidence_body(
        topic=topic,
        evid=evid,
        intent=intents["evidence"],
        analysis=intents["analysis"],
        complexity=intents["complexity"],
        knowledge=knowledge,
        mechanics=mechanics,
        time_pressure=time_pressure,
        organization=organization,
        rng=rng,
    )
    paras.extend(body)

    if intents["thesis"] == "clear" and intents["analysis"] == "causal" and time_pressure != "severe":
        if rng.random() < 0.45:
            paras.append(_closing(topic, mechanics, rng))

    if organization == "rough" and len(paras) > 2:
        # Drop a middle transition so structure feels unfinished.
        drop = 1 + rng.randrange(max(1, len(paras) - 2))
        if 0 < drop < len(paras):
            paras[drop] = paras[drop]  # keep, but merge later
    return [p for p in paras if p.strip()]


def _student_clause(text: str, mechanics: str, rng: Any) -> str:
    """Emit a clause with draft-time imperfections rather than post-polish corruption."""
    out = text.strip()
    if mechanics in {"frequent_natural_errors", "occasional_errors", "fragments_and_runons"}:
        for correct, wrong in _MISSPELL_DRAFT:
            if correct in out.lower() and rng.random() < (
                0.55 if mechanics == "frequent_natural_errors" else 0.25
            ):
                out = re.sub(re.escape(correct), wrong, out, count=1, flags=re.I)
                break
    if mechanics == "fragments_and_runons" and rng.random() < 0.18:
        # Occasional time-pressure fragment — keep rare so it does not stamp every essay.
        frag = rng.choice(
            (
                "Hard to finish in time",
                "Clock was loud by then",
                "Lost a minute rereading the prompt",
            )
        )
        out = out.rstrip(".") + ". " + frag
    return out


def _context_prior(topic: str, remembered: list[str], mechanics: str, rng: Any) -> str:
    openers = (
        f"Before the main years in the prompt, earlier patterns already mattered for {topic}.",
        f"Looking a little earlier helps, becuase older pressures shaped {topic}.",
        f"Wider background: Atlantic connections and earlier conflicts were already in play "
        f"around {_topic_slot(topic)}.",
    )
    sent = rng.choice(openers)
    if remembered and rng.random() < 0.5:
        cue = _insert_remembered(remembered[0])
        sent += " " + _student_clause(
            f"Even then, {cue} showed up in notes.",
            mechanics,
            rng,
        )
    else:
        sent += " " + _student_clause(
            "People were already arguing about land, labor, and authority.",
            mechanics,
            rng,
        )
    return sent


def _thesis_paragraph(
    prompt: str,
    topic: str,
    remembered: list[str],
    intent: str,
    cues: dict[str, Any],
    mechanics: str,
    rng: Any,
) -> str:
    evid_a = _short_label(remembered[0]) if remembered else "political pressure"
    evid_b = _short_label(remembered[1]) if len(remembered) > 1 else "social conflict"

    if intent == "unsettle":
        options = [
            f"This essay is mostly about {topic}. There were lots of factors and it is hard to pick one answer.",
            f"The prompt asks about {topic}, and different parts of the story pull in different directions.",
            f"People talk about {topic} in class but I am not sure there is one clear overall claim.",
        ]
        text = rng.choice(options)
    elif intent == "weak":
        options = [
            f"This essay is about {topic}.",
            f"There were many things happening related to {topic} in this period.",
            f"History changed in different ways when people dealt with {topic}.",
            f"The question is basically about {topic}.",
        ]
        text = rng.choice(options)
    else:
        extent = rng.choice(("a lot", "pretty clearly", "to a large extent", "in an important way"))
        options = [
            f"Overall, {topic} mattered {extent}, mainly becuase of things like {evid_a} and {evid_b}.",
            f"I think {topic} changed the period {extent}; {evid_a} pushed one way while {evid_b} reinforced it.",
            f"To a significant extent, {topic} reshaped outcomes, and examples such as {evid_a} help show why.",
        ]
        if cues.get("uses_although") and rng.random() < 0.4:
            options.append(
                f"Even though other forces mattered, {topic} still mattered {extent} through {evid_a}."
            )
        text = rng.choice(options)
        if cues.get("informal") and not text.lower().startswith("i "):
            text = "i mean, " + text[0].lower() + text[1:]
        elif cues.get("starts_lower") and text and text[0].isupper() and rng.random() < 0.5:
            text = text[0].lower() + text[1:]

    return _student_clause(text, mechanics, rng)


def _evidence_body(
    *,
    topic: str,
    evid: list[str],
    intent: str,
    analysis: str,
    complexity: str,
    knowledge: str,
    mechanics: str,
    time_pressure: str,
    organization: str,
    rng: Any,
) -> list[str]:
    paras: list[str] = []
    a = evid[0] if evid else rng.choice(_GENERIC_EVIDENCE)
    b = evid[1] if len(evid) > 1 else rng.choice([x for x in _GENERIC_EVIDENCE if x != a] or _GENERIC_EVIDENCE)
    c = evid[2] if len(evid) > 2 else None

    if intent in {"vague", "one_or_vague"}:
        paras.append(
            _student_clause(
                f"In this period people faced hard times around {topic}. Leaders made choices and "
                f"communities reacted, but it is hard to recall exact names. "
                f"Mostly there were broad trends, and maybe {_insert_remembered(a)} if I remember right.",
                mechanics,
                rng,
            )
        )
    elif intent == "names_only":
        bits = [_insert_remembered(a), _insert_remembered(b)]
        if c and knowledge == "strong":
            bits.append(_insert_remembered(c))
        paras.append(
            _student_clause(
                f"Turning to specifics, there was {bits[0]}. There was also {bits[1]}. "
                f"Notes list these next to {topic} without always saying how they prove the claim.",
                mechanics,
                rng,
            )
        )
    elif intent == "two_uneven":
        paras.append(
            _student_clause(
                f"One example is {_insert_remembered(a)}. Another is {_insert_remembered(b)}. "
                f"I know both matter for {topic}, but the connection is a little uneven in my head.",
                mechanics,
                rng,
            )
        )
        if organization == "repetitive":
            paras.append(
                _student_clause(
                    f"Again, {_insert_remembered(a)} and {_insert_remembered(b)} show up when people discuss {topic}.",
                    mechanics,
                    rng,
                )
            )
    else:  # linked
        link = rng.choice(("because", "so", "as a result", "this shows"))
        paras.append(
            _student_clause(
                f"One part of the story is {_insert_remembered(a)}, {link} it changed who held power "
                f"and how communities organized work around {topic}.",
                mechanics,
                rng,
            )
        )
        paras.append(
            _student_clause(
                f"Another angle is {_insert_remembered(b)}. Leaders and ordinary people responded "
                f"because that pressure made arguments about {topic} harder to ignore, therefore "
                f"the example actually supports the claim rather than just sitting as a name.",
                mechanics,
                rng,
            )
        )
        if c and knowledge == "strong" and time_pressure != "severe":
            paras.append(
                _student_clause(
                    f"Additional support comes from {_insert_remembered(c)}, which reinforces the same pattern.",
                    mechanics,
                    rng,
                )
            )

    if analysis == "list":
        paras.append(
            _student_clause(
                f"Also there were other developments, more changes, and more debates about {topic}. "
                f"They happened in order in my notes but I am mostly listing them.",
                mechanics,
                rng,
            )
        )
    elif analysis == "causal":
        skill_line = rng.choice(
            (
                f"Through causation, {_short_label(a)} helped produce later outcomes tied to {topic}.",
                f"By comparison, {_short_label(a)} differed from {_short_label(b)} in who benefited.",
                f"In terms of continuity and change, {_short_label(a)} marked a shift while "
                f"{_short_label(b)} shows what stayed familiar.",
            )
        )
        paras.append(_student_clause(skill_line, mechanics, rng))
        if complexity == "qualify":
            paras.append(
                _student_clause(
                    f"Still, a counterpoint is that local conditions limited how far "
                    f"{_short_label(a)} could reshape {topic}, so the change was uneven.",
                    mechanics,
                    rng,
                )
            )
    else:
        paras.append(
            _student_clause(
                f"In short, things happened around {topic} and people noticed them.",
                mechanics,
                rng,
            )
        )

    if time_pressure == "severe" and len(paras) > 2:
        paras = paras[:2]
    return paras


def _closing(topic: str, mechanics: str, rng: Any) -> str:
    return _student_clause(
        rng.choice(
            (
                f"Overall the examples keep the claim about {topic} tied to what actually happened.",
                f"Taken together, that line of reasoning about {topic} still holds for me.",
                f"So the evidence around {topic} still points the same direction in my view.",
                f"I would stick with that reading of {topic} even if details are fuzzy.",
            )
        ),
        mechanics,
        rng,
    )


def _strip_wrapper(remembered: str) -> str:
    text = remembered.strip()
    text = _MEMORY_PREFIX_RE.sub("", text)
    text = re.sub(r"\bmattering\b", "", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;.")
    return text.rstrip(".")


def _insert_remembered(remembered: str) -> str:
    """Use a remembered concept in a sentence without awkward double memory wrappers."""
    text = _strip_wrapper(remembered)
    # Full memory clauses compress to a short label so we do not embed
    # "Cotton shaped events" mid-sentence as if it were a noun phrase.
    if len(text.split()) > 5 or re.search(
        r"\b(shaped|changed|coming up|tied to|pressures|shifted|loyalty|"
        r"foreign policy|key example)\b",
        text,
        flags=re.I,
    ):
        return _short_label(remembered)
    return text


def _short_label(remembered: str) -> str:
    text = _strip_wrapper(remembered)
    entities = _extract_entity_names(text) or _extract_entity_names(remembered)
    if entities:
        # Prefer the entity whenever the stripped body is clause-like or long.
        if len(text.split()) > 4 or re.search(
            r"\b(shaped|changed|coming|tied|pressures|shifted|mattered|"
            r"loyalty|policy|example)\b",
            text,
            flags=re.I,
        ):
            return entities[0]
    words = text.split()
    if len(words) <= 6:
        return text
    if entities:
        return entities[0]
    m = re.search(r"\b([A-Z][A-Za-z0-9'’\-]+(?:\s+[A-Z][A-Za-z0-9'’\-]+){0,3})\b", text)
    if m:
        cleaned = _clean_entity_name(m.group(1))
        if cleaned:
            return cleaned
    return " ".join(words[:6]).rstrip(".,;:")


def _join_paragraphs(paragraphs: list[str], organization: str, rng: Any) -> str:
    paras = [p.strip() for p in paragraphs if p.strip()]
    if organization == "rough" and len(paras) >= 3:
        # Smash two paragraphs into one runny block.
        i = rng.randrange(len(paras) - 1)
        paras[i] = paras[i].rstrip(".") + ", and " + paras[i + 1][0].lower() + paras[i + 1][1:]
        del paras[i + 1]
    if organization == "uneven" and len(paras) >= 3:
        mid = 1
        paras[mid] = paras[mid] + " " + paras[mid + 1]
        del paras[mid + 1]
    return "\n\n".join(paras)


def _bake_mechanics_inplace(essay: str, mechanics: str, organization: str, rng: Any) -> str:
    """Reinforce draft-time voice; avoid treating this as polish-then-corrupt."""
    if mechanics == "minor_errors":
        return _maybe_misspell(essay, rng, count=1, chance=0.4)
    if mechanics == "occasional_errors":
        essay = _maybe_misspell(essay, rng, count=2, chance=0.6)
        if rng.random() < 0.35:
            essay = _join_two_sentences(essay, rng)
        return essay
    if mechanics == "frequent_natural_errors":
        essay = _maybe_misspell(essay, rng, count=3, chance=0.9)
        essay = _join_two_sentences(essay, rng)
        if rng.random() < 0.5:
            essay = _insert_fragment(essay, rng)
        return essay
    if mechanics == "fragments_and_runons":
        essay = _join_two_sentences(essay, rng)
        essay = _insert_fragment(essay, rng)
        if organization == "repetitive" and rng.random() < 0.4:
            # Repeat a short clause naturally.
            paras = essay.split("\n\n")
            if paras:
                first_words = " ".join(paras[0].split()[:8])
                paras[-1] = paras[-1] + " " + first_words + " again."
                essay = "\n\n".join(paras)
        return essay
    return essay


def _maybe_misspell(essay: str, rng: Any, *, count: int, chance: float) -> str:
    if rng.random() > chance:
        return essay
    result = essay
    applied = 0
    candidates = [(c, w) for c, w in _MISSPELL_DRAFT if c in result.lower()]
    rng.shuffle(candidates)
    for correct, wrong in candidates:
        if applied >= count:
            break
        new_result, n = re.subn(re.escape(correct), wrong, result, count=1, flags=re.I)
        if n:
            result = new_result
            applied += 1
    return result


def _join_two_sentences(essay: str, rng: Any) -> str:
    paras = essay.split("\n\n")
    if not paras:
        return essay
    idx = rng.randrange(len(paras))
    sents = re.split(r"(?<=[.!?])\s+", paras[idx].strip())
    if len(sents) < 2:
        return essay
    i = rng.randrange(len(sents) - 1)
    a = sents[i].rstrip(".")
    b = sents[i + 1]
    if b and b[0].isupper():
        b = b[0].lower() + b[1:]
    sents[i] = f"{a}, and {b}"
    del sents[i + 1]
    paras[idx] = " ".join(sents)
    return "\n\n".join(paras)


def _insert_fragment(essay: str, rng: Any) -> str:
    paras = essay.split("\n\n")
    if not paras:
        return essay
    idx = rng.randrange(len(paras))
    frag = rng.choice(
        (
            "Especially under time pressure.",
            "Not always clear from memory.",
            "At least according to class notes.",
            "Hard to measure exactly.",
            "I had to skip a detail here.",
            "That part is fuzzy in my notes.",
        )
    )
    sents = re.split(r"(?<=[.!?])\s+", paras[idx].strip())
    insert_at = rng.randrange(1, max(2, len(sents))) if len(sents) >= 2 else len(sents)
    sents.insert(insert_at, frag)
    paras[idx] = " ".join(s for s in sents if s)
    return "\n\n".join(paras)


def _fit_length(
    essay: str,
    lo: int,
    hi: int,
    topic: str,
    remembered: list[str],
    knowledge: str,
    rng: Any,
) -> str:
    words = essay.split()
    guard = 0
    while len(words) > hi and guard < 50:
        guard += 1
        paras = essay.split("\n\n")
        if len(paras) > 2:
            last = paras[-1]
            sents = re.split(r"(?<=[.!?])\s+", last.strip())
            if len(sents) > 1:
                paras[-1] = " ".join(sents[:-1])
            else:
                paras.pop()
            essay = "\n\n".join(p for p in paras if p.strip())
        else:
            sents = re.split(r"(?<=[.!?])\s+", essay.strip())
            if len(sents) > 2:
                essay = " ".join(sents[:-1])
            else:
                essay = " ".join(words[:hi])
                break
        words = essay.split()

    # Diversified pads — time-pressure / teacher / classmate lines only sometimes.
    content_pads = [
        f"I kept thinking about {topic} while writing and lost a little time.",
        f"My outline was short so the paragraphs on {topic} wander a bit.",
        f"Looking back, overlapping pressures around {topic} made one neat story hard.",
        f"I am still connecting the examples back to {topic} as I write.",
        f"Regional differences complicated any simple reading of {topic}.",
        f"From memory the timeline around {topic} jumps more than I wanted.",
    ]
    if remembered:
        content_pads.append(
            f"I tried to bring back {_short_label(remembered[0])} without copying the worksheet."
        )
    if knowledge == "strong":
        content_pads.append(
            f"A second pass would tighten how {_short_label(remembered[0]) if remembered else 'the evidence'} "
            f"supports the claim about {topic}."
        )

    rare_pads = [
        "Teachers say to use specific evidence but under the clock it gets messy.",
        "Some classmates argued politics mattered more than social change here.",
        "Hard to finish in time once the examples piled up.",
    ]
    pads = list(content_pads)
    if rng.random() < 0.3:
        pads.append(rng.choice(rare_pads))
    rng.shuffle(pads)
    pi = 0
    guard = 0
    while len(words) < lo and guard < 60:
        guard += 1
        essay = essay.rstrip() + " " + pads[pi % len(pads)]
        pi += 1
        words = essay.split()
    if len(words) > hi:
        essay = " ".join(words[:hi])
    return essay


def _cleanup_essay(
    essay: str,
    *,
    knowledge: str,
    topic: str,
    remembered: list[str],
    rng: Any,
) -> str:
    """Final authenticity hygiene: articles, mattering stubs, length/sentence floor."""
    text = essay.strip()
    # Collapse doubled articles.
    text = re.sub(r"\b([Tt]he)\s+[Tt]he\b", r"\1", text)
    text = re.sub(r"\b([Aa])\s+[Aa]\b", r"\1", text)
    text = re.sub(r"\b([Aa]n)\s+[Aa]n\b", r"\1", text)
    # Remove leftover mattering tokens and tidy whitespace/punctuation.
    text = re.sub(r"\bmattering\b", "", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?]){2,}", r"\1", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Ensure multiple sentences.
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(sents) < 2:
        extra = rng.choice(
            (
                f"That still leaves open questions about {topic}.",
                "I would add one more example if the clock allowed.",
                f"The notes on {topic} were thinner than I wanted.",
            )
        )
        text = text.rstrip() + " " + extra
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    # Competent+ essays should clear ~100 words after hygiene.
    if knowledge in {"competent", "strong"}:
        words = text.split()
        guard = 0
        fillers = [
            f"I am still tying the examples back to {topic}.",
            f"Regional fights made {topic} harder to summarize cleanly.",
            (
                f"Even a quick mention of {_short_label(remembered[0])} helps anchor the claim."
                if remembered
                else f"Even a quick mention of debates in Congress helps anchor the claim about {topic}."
            ),
            f"Under the time limit the analysis of {topic} stays a little uneven.",
        ]
        fi = 0
        while len(words) < 100 and guard < 20:
            guard += 1
            text = text.rstrip() + " " + fillers[fi % len(fillers)]
            fi += 1
            words = text.split()

    # One more article / mattering pass after padding.
    text = re.sub(r"\b([Tt]he)\s+[Tt]he\b", r"\1", text)
    text = re.sub(r"\b([Aa])\s+[Aa]\b", r"\1", text)
    text = re.sub(r"\bmattering\b", "", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _scrub_eight_gram_copies(
    essay: str,
    cards: Sequence[Mapping[str, Any]],
    style_ref: str,
    rng: Any,
) -> str:
    sources = [str(c.get("concept") or c.get("fact") or "") for c in cards]
    if style_ref.strip():
        sources.append(style_ref[:400])
    sources = [s for s in sources if s.strip()]
    if not sources:
        return essay

    result = essay
    for _ in range(4):
        hit = False
        for source in sources:
            if not has_eight_gram_overlap(result, source):
                continue
            hit = True
            # Break overlapping region by deleting a mid-window of shared words.
            result = _break_overlap(result, source, rng)
        if not hit:
            break
    return result


def _break_overlap(essay: str, source: str, rng: Any) -> str:
    essay_words = normalize_essay(essay).split()
    source_grams = set()
    src_words = normalize_essay(source).split()
    for i in range(max(0, len(src_words) - 7)):
        source_grams.add(tuple(src_words[i : i + 8]))
    raw_words = essay.split()
    # Map normalized tokens back approximately by index on raw split.
    norm_to_raw = []
    for w in raw_words:
        nw = normalize_essay(w)
        if nw:
            norm_to_raw.append(w)
    # Find first overlapping 8-gram in essay.
    for i in range(max(0, len(essay_words) - 7)):
        gram = tuple(essay_words[i : i + 8])
        if gram in source_grams:
            # Replace middle tokens with studenty paraphrase glue.
            glue = rng.choice(("kind of", "basically", "from what I remember"))
            # Operate on raw_words if lengths align closely; else soft rewrite whole span.
            if len(norm_to_raw) == len(essay_words) and i + 8 <= len(norm_to_raw):
                norm_to_raw[i + 3] = glue
                norm_to_raw[i + 4] = "about"
                return " ".join(norm_to_raw)
            # Fallback: drop a chunk of the essay near the overlap zone.
            drop_start = max(0, int(len(raw_words) * (i / max(1, len(essay_words)))) )
            for j in range(drop_start, min(len(raw_words), drop_start + 4)):
                if j < len(raw_words):
                    raw_words[j] = glue if j == drop_start else ""
            return " ".join(w for w in raw_words if w)
    # Last resort: shuffle a mid sentence.
    paras = essay.split("\n\n")
    if paras:
        words = paras[0].split()
        if len(words) > 10:
            words[4:8] = ["from", "memory", "it", "was"]
            paras[0] = " ".join(words)
        return "\n\n".join(paras)
    return essay
