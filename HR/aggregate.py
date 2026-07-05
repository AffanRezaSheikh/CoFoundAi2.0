"""
Aggregate per-question scores from score.py into a candidate-level verdict.

Input shape (per_question entry from score.py):
    {
        "question_id": "q1",
        "skill_tested": "Python",
        "technical_accuracy": 7, "depth": 6, "relevance": 8, "clarity": 7,
        "ai_likeness": 3, "red_flags": [...], "feedback": "...",
        "keyword_hits": [...], "aggregate": 6.5,
    }

Output: a recruiter-facing verdict with breakdowns by skill/difficulty
plus a hire recommendation grounded in coverage, not just raw average.
"""

import logging
import statistics
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

# Weights for the overall score. ai_likeness is handled separately as a penalty.
DIMENSION_WEIGHTS = {
    "technical_accuracy": 0.35,
    "depth": 0.25,
    "relevance": 0.25,
    "clarity": 0.15,
}

# A question is "skipped" if it has any of these flags — separates true 0s
# (bad answer) from non-answers (silence, transcription failure, no audio).
SKIP_FLAGS = {"no_answer_provided", "silence_or_unintelligible", "transcription_failed"}

HIGH_AI_THRESHOLD = 7  # ai_likeness above this is a per-question concern

# Recommendation thresholds — tuned for the [0, 10] scale.
# Coverage = fraction of questions actually scored (not skipped).
RECOMMENDATION_THRESHOLDS = [
    # (label, min_overall, max_red_flags, max_high_ai, min_coverage)
    ("strong_hire", 7.5, 1, 0, 0.8),
    ("hire",       6.0, 2, 1, 0.7),
    ("maybe",      4.5, 3, 2, 0.5),
]


# ─────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────

def aggregate_candidate_scores(
    question_scores: List[Dict],
    questions: Optional[List[Dict]] = None,
) -> Dict:
    """
    Build a candidate-level verdict from per-question scores.

    Args:
        question_scores: list from score.py's per_question output.
        questions: optional original question list — enables per-difficulty
                   and per-category breakdowns (those fields live on the
                   question, not the score).
    """
    if not question_scores:
        return _empty_verdict()

    scored, skipped = _split_scored_skipped(question_scores)

    if not scored:
        # Every answer was a non-answer
        verdict = _empty_verdict()
        verdict["coverage"] = 0.0
        verdict["skipped_count"] = len(skipped)
        verdict["total_questions"] = len(question_scores)
        verdict["recommendation"] = "reject"
        verdict["recommendation_reason"] = "no answerable responses"
        return verdict

    avg_scores = _average_dimensions(scored)
    overall = _weighted_overall(avg_scores)
    technical = (avg_scores["technical_accuracy"] + avg_scores["depth"]) / 2
    communication = (avg_scores["relevance"] + avg_scores["clarity"]) / 2

    high_ai_count = sum(1 for s in scored if s.get("ai_likeness", 0) > HIGH_AI_THRESHOLD)
    ai_risk = _ai_risk_level(avg_scores["ai_likeness"], high_ai_count, len(scored))

    all_flags = [f for s in scored for f in s.get("red_flags", [])]
    unique_flags = sorted(set(all_flags))
    coverage = len(scored) / len(question_scores)
    consistency = _consistency_score(scored)

    recommendation, reason = _recommend(
        overall=overall,
        red_flag_count=len(unique_flags),
        high_ai_count=high_ai_count,
        coverage=coverage,
        ai_risk=ai_risk,
        consistency=consistency,
    )

    return {
        "overall_score": round(overall, 2),
        "technical_score": round(technical, 2),
        "communication_score": round(communication, 2),
        "ai_suspicion_level": round(avg_scores["ai_likeness"], 2),
        "ai_risk": ai_risk,
        "high_ai_count": high_ai_count,
        "red_flag_count": len(unique_flags),
        "red_flags": unique_flags,
        "score_breakdown": {k: round(v, 2) for k, v in avg_scores.items()},
        "by_skill": _breakdown_by_field(scored, "skill_tested"),
        "by_difficulty": _breakdown_by_question_field(scored, questions, "difficulty"),
        "by_category": _breakdown_by_question_field(scored, questions, "category"),
        "coverage": round(coverage, 2),
        "consistency": round(consistency, 2),
        "scored_count": len(scored),
        "skipped_count": len(skipped),
        "total_questions": len(question_scores),
        "recommendation": recommendation,
        "recommendation_reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────
# CORE MATH
# ─────────────────────────────────────────────────────────────────────

def _split_scored_skipped(question_scores: List[Dict]):
    scored, skipped = [], []
    for s in question_scores:
        flags = set(s.get("red_flags", []))
        if flags & SKIP_FLAGS:
            skipped.append(s)
        else:
            scored.append(s)
    return scored, skipped


def _average_dimensions(scored: List[Dict]) -> Dict[str, float]:
    dims = list(DIMENSION_WEIGHTS) + ["ai_likeness"]
    return {
        dim: statistics.fmean(s.get(dim, 0) for s in scored)
        for dim in dims
    }


def _weighted_overall(avg_scores: Dict[str, float]) -> float:
    positive = sum(avg_scores[dim] * w for dim, w in DIMENSION_WEIGHTS.items())
    ai_penalty = (avg_scores["ai_likeness"] / 10.0) * 2.0  # max 2 points off
    return max(0.0, positive - ai_penalty)


def _consistency_score(scored: List[Dict]) -> float:
    """
    1.0 = uniform performance, 0.0 = wildly inconsistent.
    Useful signal: a candidate who alternates 9s and 1s is different from
    one who scores 5s across the board even if the mean is the same.
    """
    aggregates = [s.get("aggregate") for s in scored if "aggregate" in s]
    if len(aggregates) < 2:
        return 1.0
    stdev = statistics.stdev(aggregates)
    # Map stdev (typical 0..5 on a 0-10 scale) to consistency 1..0
    return max(0.0, 1.0 - (stdev / 5.0))


def _ai_risk_level(avg_ai: float, high_ai_count: int, total: int) -> str:
    if avg_ai >= 7 or high_ai_count >= max(2, total // 2):
        return "high"
    if avg_ai >= 5 or high_ai_count >= 1:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────
# BREAKDOWNS
# ─────────────────────────────────────────────────────────────────────

def _breakdown_by_field(scored: List[Dict], field: str) -> Dict[str, Dict]:
    """Group scores by a field present on each score entry (e.g. skill_tested)."""
    buckets: Dict[str, List[Dict]] = {}
    for s in scored:
        key = s.get(field) or "unspecified"
        buckets.setdefault(key, []).append(s)

    return {
        key: _bucket_summary(items)
        for key, items in buckets.items()
    }


def _breakdown_by_question_field(
    scored: List[Dict],
    questions: Optional[List[Dict]],
    field: str,
) -> Dict[str, Dict]:
    """
    Group by a field that lives on the question (difficulty, category) by
    matching question_id back to the questions list. Returns {} if questions
    were not provided.
    """
    if not questions:
        return {}

    # questions are positional → q_index = qid number - 1
    by_id = {f"q{i+1}": q for i, q in enumerate(questions)}
    buckets: Dict[str, List[Dict]] = {}
    for s in scored:
        q = by_id.get(s.get("question_id", ""))
        if not q:
            continue
        key = q.get(field) or "unspecified"
        buckets.setdefault(key, []).append(s)

    return {
        key: _bucket_summary(items)
        for key, items in buckets.items()
    }


def _bucket_summary(items: List[Dict]) -> Dict:
    aggregates = [s.get("aggregate", 0.0) for s in items]
    return {
        "count": len(items),
        "avg_aggregate": round(statistics.fmean(aggregates), 2) if aggregates else 0.0,
        "avg_technical": round(statistics.fmean(s.get("technical_accuracy", 0) for s in items), 2),
        "avg_clarity": round(statistics.fmean(s.get("clarity", 0) for s in items), 2),
    }


# ─────────────────────────────────────────────────────────────────────
# RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────

def _recommend(
    overall: float,
    red_flag_count: int,
    high_ai_count: int,
    coverage: float,
    ai_risk: str = "low",
    consistency: float = 1.0,
):
    # Hard veto: aggregate AI suspicion is high even if no single answer crossed
    # the per-question threshold (e.g. all answers right at ai_likeness=7).
    if ai_risk == "high":
        return "reject", "high likelihood of AI-generated answers (avg ai_likeness elevated)"

    label = "reject"
    for tier, min_overall, max_flags, max_ai, min_cov in RECOMMENDATION_THRESHOLDS:
        if (
            overall >= min_overall
            and red_flag_count <= max_flags
            and high_ai_count <= max_ai
            and coverage >= min_cov
        ):
            label = tier
            break

    # Demotion guard: very inconsistent performance shouldn't auto-promote past 'maybe'
    if consistency < 0.4 and label in ("strong_hire", "hire"):
        label = "maybe"
        return label, _reason(label, overall, red_flag_count, high_ai_count, coverage) + f", consistency={consistency:.2f} (demoted)"

    return label, _reason(label, overall, red_flag_count, high_ai_count, coverage)


def _reason(label: str, overall: float, flags: int, high_ai: int, coverage: float) -> str:
    parts = [f"overall={overall:.1f}", f"flags={flags}", f"high_ai={high_ai}", f"coverage={coverage:.0%}"]
    return f"{label}: " + ", ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# EMPTY / EDGE CASE
# ─────────────────────────────────────────────────────────────────────

def _empty_verdict() -> Dict:
    return {
        "overall_score": 0.0,
        "technical_score": 0.0,
        "communication_score": 0.0,
        "ai_suspicion_level": 0.0,
        "ai_risk": "unknown",
        "high_ai_count": 0,
        "red_flag_count": 0,
        "red_flags": [],
        "score_breakdown": {},
        "by_skill": {},
        "by_difficulty": {},
        "by_category": {},
        "coverage": 0.0,
        "consistency": 0.0,
        "scored_count": 0,
        "skipped_count": 0,
        "total_questions": 0,
        "recommendation": "reject",
        "recommendation_reason": "no scores provided",
    }
