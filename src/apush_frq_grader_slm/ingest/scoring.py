"""Score helpers for third-party sources with total-only labels."""

from __future__ import annotations

TOTAL_TO_ROWS: dict[int, tuple[int, int, int, int]] = {
    6: (1, 1, 2, 2),
    5: (1, 1, 2, 1),
    4: (1, 1, 1, 1),
    3: (1, 0, 1, 1),
    2: (0, 0, 1, 0),
    1: (0, 0, 1, 0),
    0: (0, 0, 0, 0),
}


def total_to_row_scores(total: int) -> dict[str, int]:
    """Map a 0–6 total to a conservative row score split."""
    clamped = max(0, min(6, int(total)))
    thesis, contextualization, evidence, analysis_reasoning = TOTAL_TO_ROWS.get(
        clamped, (0, 0, 0, 0)
    )
    return {
        "thesis": thesis,
        "contextualization": contextualization,
        "evidence": evidence,
        "analysis_reasoning": analysis_reasoning,
    }
