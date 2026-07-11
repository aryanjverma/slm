"""Deterministic AMSCO-grounded LEQ essay composer for v4 fill/backup.

Produces unique, historically plausible student essays whose structure matches
the task's target rubric profile (thesis / contextualization / evidence /
analysis_reasoning). Knowledge comes from an AMSCO ``kb_bundle`` as returned by
``knowledge.amsco.facts_for_prompt``.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any

# Detectable strong-claim markers used by thesis=1 templates (and unit tests).
STRONG_CLAIM_RE = re.compile(
    r"\b("
    r"to a (?:significant|large|great|considerable|limited|moderate) extent|"
    r"this essay argues that|"
    r"a historically defensible claim is that|"
    r"the evidence supports the claim that"
    r")\b",
    re.IGNORECASE,
)

_PAD_SENTENCES = (
    "Teachers often remind students to stay focused on the prompt timeline.",
    "Classroom notes from earlier units also mentioned related developments.",
    "Some classmates debated how much weight to give political versus social change.",
    "Looking back, the period feels crowded with overlapping pressures.",
    "It is easy to lose track of which decade each example belongs to.",
    "Review sheets listed several names without explaining why they mattered.",
    "A short outline helped keep the paragraphs from wandering too far.",
    "Time pressure made it hard to polish every transition.",
)

_MISSPELLINGS: tuple[tuple[str, str], ...] = (
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
)


def rng_for_task(task_id: str, salt: int = 0) -> random.Random:
    """Build a deterministic RNG from ``task_id`` (and optional salt)."""
    digest = hashlib.sha256(f"{task_id}:{salt}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def compose_essay(task: dict[str, Any], kb_bundle: dict[str, Any], rng: random.Random) -> str:
    """Compose one student essay matching ``task['target_scores']`` and persona."""
    prompt = str(task.get("prompt") or "")
    scores = dict(task.get("target_scores") or {})
    thesis = int(scores.get("thesis", 0))
    contextualization = int(scores.get("contextualization", 0))
    evidence = int(scores.get("evidence", 0))
    analysis = int(scores.get("analysis_reasoning", 0))
    persona = dict(task.get("persona") or {})
    skill = str(task.get("reasoning_skill") or "causation")
    band = task.get("length_band") or [140, 220]
    lo, hi = int(band[0]), int(band[1])

    knowledge = str(persona.get("historical_knowledge", "competent"))
    mechanics = str(persona.get("mechanics", "clean"))
    misconception = str(persona.get("misconception", "none"))

    hooks = list(kb_bundle.get("context_hooks") or [])
    facts = list(kb_bundle.get("key_facts") or [])
    bank = list(kb_bundle.get("evidence_bank") or [])
    wrong = list(kb_bundle.get("misconceptions") or [])

    rng.shuffle(hooks)
    rng.shuffle(facts)
    rng.shuffle(bank)
    rng.shuffle(wrong)

    keep = {"weak": 3, "uneven": 6, "competent": 10, "strong": 16}.get(knowledge, 8)
    facts = facts[:keep]
    bank = bank[: max(keep, 4)]
    if knowledge == "weak":
        bank = bank[:3]
        facts = facts[:2]

    topic = _topic_phrase(prompt)
    evid_a, evid_b, evid_extra = _pick_evidence_names(bank, facts, rng)

    paragraphs: list[str] = []

    if contextualization == 1:
        paragraphs.append(_context_paragraph(hooks, facts, topic, rng))

    # Avoid leaking named bank items when evidence target is 0.
    thesis_a = evid_a if evidence >= 1 else "political pressure"
    thesis_b = evid_b if evidence >= 1 else "social conflict"

    if thesis == 1:
        paragraphs.append(_strong_thesis(topic, skill, thesis_a, thesis_b, rng))
    else:
        paragraphs.append(_weak_thesis(prompt, topic, rng))

    body = _body_paragraphs(
        evidence=evidence,
        analysis=analysis,
        skill=skill,
        topic=topic,
        evid_a=evid_a,
        evid_b=evid_b,
        evid_extra=evid_extra,
        facts=facts,
        misconception=misconception,
        wrong=wrong,
        knowledge=knowledge,
        rng=rng,
    )
    paragraphs.extend(body)

    if thesis == 1 and analysis >= 1 and rng.random() < 0.55:
        paragraphs.append(_closing_restatement(topic, skill, rng))

    if mechanics == "uneven_paragraphs" and len(paragraphs) >= 3:
        # Merge a short middle paragraph into its neighbor for uneven shape.
        mid = 1 + (rng.randrange(len(paragraphs) - 2) if len(paragraphs) > 2 else 0)
        if mid < len(paragraphs) - 1:
            paragraphs[mid] = paragraphs[mid] + " " + paragraphs[mid + 1]
            del paragraphs[mid + 1]

    essay = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    essay = _apply_mechanics(essay, mechanics, rng)
    essay = _fit_length(essay, lo, hi, topic, evid_a, evid_b, rng)
    return essay.strip()


def _topic_phrase(prompt: str) -> str:
    text = prompt.strip()
    text = re.sub(
        r"^(Evaluate|Analyze|Compare|Explain)\s+(the extent to which|how|why|whether)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).rstrip(".")
    if len(text) > 120:
        text = text[:117].rsplit(" ", 1)[0] + "…"
    return text[0].lower() + text[1:] if text else "this historical development"


def _pick_evidence_names(
    bank: list[str],
    facts: list[str],
    rng: random.Random,
) -> tuple[str, str, list[str]]:
    names = [str(x).strip() for x in bank if str(x).strip()]
    if len(names) < 2:
        # Fall back to short noun phrases carved from key_facts.
        for fact in facts:
            snippet = _short_fact_label(fact)
            if snippet and snippet not in names:
                names.append(snippet)
            if len(names) >= 4:
                break
    while len(names) < 2:
        names.append(rng.choice(["congressional debates", "reform societies", "court rulings"]))
    rng.shuffle(names)
    return names[0], names[1], names[2:6]


def _short_fact_label(fact: str) -> str:
    # Prefer a capitalized multi-word span; else first ~6 words.
    m = re.search(r"\b([A-Z][A-Za-z0-9'’\-]+(?:\s+[A-Z][A-Za-z0-9'’\-]+){0,3})\b", fact)
    if m and len(m.group(1)) >= 4:
        return m.group(1)
    words = fact.split()
    return " ".join(words[:6]).rstrip(".,;:")


def _context_paragraph(
    hooks: list[str],
    facts: list[str],
    topic: str,
    rng: random.Random,
) -> str:
    openings = (
        "Before the main developments in the prompt period, broader patterns were already visible.",
        "To place the question in a wider frame, earlier conditions matter.",
        "Looking just before the years in the prompt helps explain later pressure.",
        "Broader background helps explain later change across regions.",
    )
    sentences: list[str] = [rng.choice(openings)]
    chosen_hooks = hooks[: rng.randint(2, 3)] if hooks else []
    for hook in chosen_hooks:
        sentences.append(str(hook).rstrip(".") + ".")
    if len(sentences) < 3 and facts:
        sentences.append(str(facts[0]).rstrip(".") + ".")
    while len(sentences) < 3:
        sentences.append(
            rng.choice(
                (
                    f"Earlier political and economic patterns already shaped debates about {topic}.",
                    "Prior wars, migrations, and institutions set expectations for what came next.",
                    "Older regional rivalries and Atlantic connections still influenced daily life.",
                )
            )
        )
    # 3–4 sentences total.
    if len(sentences) < 4 and (hooks[3:] if len(hooks) > 3 else facts[1:]):
        extra = (hooks[3:] or facts[1:])[0]
        sentences.append(str(extra).rstrip(".") + ".")
    elif len(sentences) < 4:
        sentences.append(
            f"Those earlier conditions made later arguments about {topic} feel urgent."
        )
    return " ".join(sentences[:4])


def _strong_thesis(
    topic: str,
    skill: str,
    evid_a: str,
    evid_b: str,
    rng: random.Random,
) -> str:
    extent = rng.choice(
        ("significant", "large", "considerable", "limited", "moderate")
    )
    templates = [
        (
            f"To a {extent} extent, {topic}, because developments such as {evid_a} "
            f"and {evid_b} created a clear line of reasoning about change over time."
        ),
        (
            f"This essay argues that {topic} mattered to a {extent} extent, "
            f"since {evid_a} pushed one set of outcomes while {evid_b} reinforced them."
        ),
        (
            f"A historically defensible claim is that {topic} reshaped the period "
            f"to a {extent} extent; {evid_a} and {evid_b} show why that claim holds."
        ),
        (
            f"The evidence supports the claim that {topic} was {extent} in its effects, "
            f"because {evid_a} altered incentives and {evid_b} locked those changes in place."
        ),
    ]
    if skill == "comparison":
        templates.append(
            f"To a {extent} extent, comparing forces behind {topic} shows that "
            f"{evid_a} mattered more than {evid_b}, although both shaped outcomes."
        )
    elif skill == "ccot":
        templates.append(
            f"To a {extent} extent, {topic} reveals continuity and change: "
            f"{evid_a} marked a shift while {evid_b} preserved older patterns."
        )
    return rng.choice(templates)


def _weak_thesis(prompt: str, topic: str, rng: random.Random) -> str:
    restatements = [
        f"This essay is about {topic}.",
        f"The prompt asks students to discuss {topic}.",
        f"There were many things happening related to {topic} in this period.",
        f"History changed in different ways when people dealt with {topic}.",
        f"In this time period, {topic} was a topic people talked about.",
        f"The question is: {prompt.rstrip('.')}.",
    ]
    return rng.choice(restatements)


def _body_paragraphs(
    *,
    evidence: int,
    analysis: int,
    skill: str,
    topic: str,
    evid_a: str,
    evid_b: str,
    evid_extra: list[str],
    facts: list[str],
    misconception: str,
    wrong: list[str],
    knowledge: str,
    rng: random.Random,
) -> list[str]:
    openings = (
        "In the body of the response,",
        "Turning to specific developments,",
        "One part of the story involves",
        "Another angle comes from",
        "During these years,",
        "Students often remember that",
    )
    paras: list[str] = []

    if evidence == 0:
        vague = [
            (
                f"{rng.choice(openings)} people faced hard times and leaders made choices. "
                f"Many groups wanted change, and some policies seemed important. "
                f"Overall things shifted in society without clear named examples."
            ),
            (
                f"Life was complicated around {topic}. Communities reacted in different ways, "
                f"and politics sometimes followed economic pressure. "
                f"It is difficult to list exact laws or people from memory."
            ),
        ]
        paras.append(rng.choice(vague))
        if misconception != "none" and wrong and knowledge in {"weak", "uneven"}:
            # Misconception text can contain proper nouns; keep it mild and generic.
            paras[-1] += " Some remembered details may be mixed up across decades."
        if analysis >= 1:
            paras.append(_analysis_only_paragraph(skill, topic, analysis, rng))
        return paras

    if evidence == 1:
        # Name-drop ≥2 specific items without because/therefore linkage to thesis.
        extras = ", ".join(evid_extra[:2]) if evid_extra else "related reforms"
        paras.append(
            f"{rng.choice(openings)} {evid_a}. There was also {evid_b}. "
            f"Notes also mention {extras}. These names appear in textbooks about the era."
        )
        if facts:
            paras.append(
                f"A worksheet listed a detail: {facts[0].rstrip('.')}. "
                f"Another bullet simply named {evid_a} again next to {evid_b}."
            )
        else:
            paras.append(
                f"Class notes also listed {evid_a} and {evid_b} without explaining causation."
            )
        if misconception != "none" and wrong and knowledge in {"weak", "uneven"}:
            paras[-1] += " " + _misconception_sentence(wrong, rng)
        if analysis >= 1:
            paras.append(_analysis_only_paragraph(skill, topic, analysis, rng))
        return paras

    # evidence == 2: use ≥2 specific examples with because/therefore linkage.
    link_words = ("because", "therefore", "as a result", "this shows that")
    link1 = rng.choice(link_words)
    link2 = rng.choice([w for w in link_words if w != link1] or link_words)
    fact_bit = ""
    if facts and knowledge in {"competent", "strong"}:
        fact_bit = f" {facts[0].rstrip('.')}."

    link2_clause = {
        "because": "Because of that linkage, the thesis holds:",
        "therefore": "Therefore the thesis holds:",
        "as a result": "As a result, the thesis holds:",
        "this shows that": "This shows that the thesis holds:",
    }.get(link2, "Therefore the thesis holds:")
    para1 = (
        f"{rng.choice(openings)} {evid_a} mattered for {topic} {link1} it changed "
        f"who held power and how communities organized work.{fact_bit} "
        f"{link2_clause} specific evidence like {evid_a} actively supports the claim "
        f"rather than sitting as a disconnected name."
    )

    para2 = (
        f"{rng.choice(openings)} {evid_b}. Leaders and ordinary people responded "
        f"because {evid_b} created new pressures; therefore arguments about {topic} "
        f"became harder to ignore. Together, {evid_a} and {evid_b} prove the claim "
        f"by linking concrete examples to the line of reasoning."
    )
    if evid_extra and knowledge == "strong":
        para2 += f" Additional support comes from {evid_extra[0]}, which reinforces the same pattern."

    if misconception != "none" and wrong and knowledge == "weak":
        para2 += " " + _misconception_sentence(wrong, rng)

    paras.extend([para1, para2])

    if analysis >= 1:
        paras.append(_reasoning_paragraph(skill, topic, evid_a, evid_b, analysis, rng))
    elif analysis == 0:
        # Keep descriptive-only tone: strip reasoning paragraph; body already has
        # because/therefore for evidence=2 (needed for evidence point). Add a flat
        # descriptive closer without causation/comparison/CCOT framing.
        paras.append(
            f"In short, {evid_a} and {evid_b} were present in the period and people "
            f"noticed them. Descriptions of events filled notebooks, but the essay "
            f"mostly retells what happened around {topic}."
        )
    return paras


def _misconception_sentence(wrong: list[str], rng: random.Random) -> str:
    item = rng.choice(wrong)
    # Soften into a student-voiced mild confusion (do not label as wrong).
    short = item if len(item) < 160 else item[:157].rsplit(" ", 1)[0] + "…"
    return rng.choice(
        (
            f"Some students also think that {short[0].lower() + short[1:]}",
            f"It seems possible that {short[0].lower() + short[1:]}",
            f"A confusing memory is that {short[0].lower() + short[1:]}",
        )
    )


def _analysis_skill_language(skill: str, topic: str, evid_a: str, evid_b: str) -> str:
    if skill == "comparison":
        return (
            f"By comparison, {evid_a} differed from {evid_b} in who benefited, "
            f"even though both connected to {topic}."
        )
    if skill == "ccot":
        return (
            f"In terms of continuity and change, {evid_a} marked a shift while "
            f"{evid_b} shows what stayed familiar within {topic}."
        )
    if skill == "relative_importance":
        return (
            f"Weighing relative importance, {evid_a} was more decisive than {evid_b} "
            f"in driving {topic}, although both contributed."
        )
    # causation / extent default
    return (
        f"Through causation, {evid_a} helped produce later outcomes tied to {topic}, "
        f"and {evid_b} intensified those effects."
    )


def _analysis_only_paragraph(
    skill: str,
    topic: str,
    analysis: int,
    rng: random.Random,
) -> str:
    # Used when evidence < 2 but analysis >= 1: structure with skill language,
    # optionally without naming bank items (evidence=0) — use generic nouns.
    evid_a = "one major development"
    evid_b = "another related pressure"
    base = _analysis_skill_language(skill, topic, evid_a, evid_b)
    if analysis >= 2:
        base += " " + _complexity_clause(topic, evid_a, evid_b, rng)
    return base


def _reasoning_paragraph(
    skill: str,
    topic: str,
    evid_a: str,
    evid_b: str,
    analysis: int,
    rng: random.Random,
) -> str:
    base = _analysis_skill_language(skill, topic, evid_a, evid_b)
    if analysis >= 2:
        base += " " + _complexity_clause(topic, evid_a, evid_b, rng)
    return base


def _complexity_clause(topic: str, evid_a: str, evid_b: str, rng: random.Random) -> str:
    options = [
        (
            f"Still, a counterargument is that local conditions limited how far "
            f"{evid_a} could reshape {topic}, so the change was uneven."
        ),
        (
            f"Nuance matters: while {evid_a} supports the claim, critics could point "
            f"to {evid_b} as evidence that older patterns persisted."
        ),
        (
            f"From multiple perspectives, reformers, officials, and ordinary families "
            f"experienced {topic} differently, which complicates a single story."
        ),
        (
            f"Although the main claim stands, acknowledging opposing views about "
            f"{evid_b} shows complexity rather than a one-sided narrative."
        ),
    ]
    return rng.choice(options)


def _closing_restatement(topic: str, skill: str, rng: random.Random) -> str:
    return rng.choice(
        [
            f"In conclusion, the line of reasoning about {topic} remains historically defensible.",
            f"Overall, structuring the argument around {skill} keeps the claim tied to the evidence.",
            f"Taken together, the examples sustain a clear claim regarding {topic}.",
        ]
    )


def _apply_mechanics(essay: str, mechanics: str, rng: random.Random) -> str:
    if mechanics in {"clean", "uneven_paragraphs"}:
        return essay
    if mechanics == "run_ons":
        return _inject_run_ons(essay, rng)
    if mechanics == "fragments":
        return _inject_fragments(essay, rng)
    if mechanics == "misspellings":
        return _inject_misspellings(essay, rng, count=3)
    if mechanics == "ordinary_errors":
        essay = _inject_misspellings(essay, rng, count=1)
        if rng.random() < 0.5:
            essay = _inject_run_ons(essay, rng, max_joins=1)
        return essay
    return essay


def _inject_run_ons(essay: str, rng: random.Random, max_joins: int = 2) -> str:
    parts = essay.split("\n\n")
    joins = 0
    out: list[str] = []
    for para in parts:
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        if len(sentences) >= 2 and joins < max_joins and rng.random() < 0.7:
            i = rng.randrange(len(sentences) - 1)
            a = sentences[i].rstrip(".")
            b = sentences[i + 1]
            if b and b[0].isupper():
                b = b[0].lower() + b[1:]
            sentences[i] = f"{a}, and {b}"
            del sentences[i + 1]
            joins += 1
        out.append(" ".join(s for s in sentences if s))
    return "\n\n".join(out)


def _inject_fragments(essay: str, rng: random.Random) -> str:
    parts = essay.split("\n\n")
    if not parts:
        return essay
    idx = rng.randrange(len(parts))
    sentences = re.split(r"(?<=[.!?])\s+", parts[idx].strip())
    if len(sentences) >= 2:
        frag = rng.choice(
            (
                "Especially in crowded cities.",
                "Not always clear in the sources.",
                "Hard to measure precisely.",
                "At least according to class notes.",
            )
        )
        insert_at = rng.randrange(1, len(sentences))
        sentences.insert(insert_at, frag)
        parts[idx] = " ".join(sentences)
    return "\n\n".join(parts)


def _inject_misspellings(essay: str, rng: random.Random, count: int = 3) -> str:
    candidates = [(correct, wrong) for correct, wrong in _MISSPELLINGS if correct in essay.lower()]
    if not candidates:
        return essay
    rng.shuffle(candidates)
    result = essay
    applied = 0
    for correct, wrong in candidates:
        if applied >= count:
            break
        pattern = re.compile(re.escape(correct), re.IGNORECASE)

        def _repl(match: re.Match[str], w: str = wrong) -> str:
            src = match.group(0)
            if src.isupper():
                return w.upper()
            if src[0].isupper():
                return w.capitalize()
            return w

        new_result, n = pattern.subn(_repl, result, count=1)
        if n:
            result = new_result
            applied += 1
    return result


def _fit_length(
    essay: str,
    lo: int,
    hi: int,
    topic: str,
    evid_a: str,
    evid_b: str,
    rng: random.Random,
) -> str:
    words = essay.split()
    # Trim from the end if too long (prefer dropping pad / last sentences).
    guard = 0
    while len(words) > hi and guard < 40:
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

    # Pad if too short.
    pad_pool = list(_PAD_SENTENCES)
    rng.shuffle(pad_pool)
    extras = [
        f"Additional classroom discussion about {topic} filled leftover minutes.",
        f"A quick reminder of {evid_a} and {evid_b} helped meet the length expectation.",
        f"Some students repeated that {topic} connected politics, economy, and culture.",
    ]
    pad_pool = extras + pad_pool
    pi = 0
    guard = 0
    while len(words) < lo and guard < 60:
        guard += 1
        addition = pad_pool[pi % len(pad_pool)]
        pi += 1
        essay = essay.rstrip() + " " + addition
        words = essay.split()
    # If still slightly over after pad oscillation, soft trim.
    if len(words) > hi:
        essay = " ".join(words[:hi])
    return essay
