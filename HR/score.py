"""
Score interview answers using Gemini.

Consumes:
  - questions from generate_question.py (fields: question, skill_tested,
    expected_keywords, difficulty, ...)
  - transcripts from transcriptions_pro.py (dict with text/duration/error/...)

Produces a per-question score dict and a session-level aggregate.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SCORE_DIMENSIONS = ["technical_accuracy", "depth", "relevance", "clarity", "ai_likeness"]

# Aggregate weights — ai_likeness is a penalty, not a positive dimension
AGGREGATE_WEIGHTS = {
    "technical_accuracy": 0.35,
    "depth": 0.25,
    "relevance": 0.25,
    "clarity": 0.15,
}

MIN_TRANSCRIPT_CHARS = 5  # below this, treat as silence
MAX_TRANSCRIPT_CHARS = 4000  # ~1000 tokens — caps cost/latency on long answers


class AnswerScorer:
    """
    Score interview answers using Gemini with deterministic settings.

    Usage:
        scorer = AnswerScorer()
        result = scorer.score_answer(question, transcript_dict)
        session = scorer.score_session(questions, transcripts)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file")

        genai.configure(api_key=self.api_key)
        # Low temperature → consistent grading across re-runs
        self.model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"temperature": 0.2, "max_output_tokens": 800},
        )

    # ─────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────

    def score_answer(
        self,
        question: Dict,
        transcript: Optional[Union[Dict, str]],
    ) -> Dict:
        """
        Score a single answer.

        Accepts either a string transcript or the full dict from
        transcriptions_pro.py — the dict carries error/duration info that
        helps us distinguish "no answer", "silence", and "transcription failed".
        """
        text, status = self._unpack_transcript(transcript)

        if status == "no_answer":
            return self._no_answer_score()
        if status == "transcription_failed":
            return self._transcription_failed_score()
        if status == "silence":
            return self._silence_score()

        # Cap transcript length to bound prompt cost/latency
        if len(text) > MAX_TRANSCRIPT_CHARS:
            logger.info("Truncating transcript: %d → %d chars", len(text), MAX_TRANSCRIPT_CHARS)
            text = text[:MAX_TRANSCRIPT_CHARS]

        # Cheap deterministic baseline before paying for the LLM call
        keyword_hits = self._keyword_overlap(question, text)

        prompt = self._build_scoring_prompt(question, text, keyword_hits)
        raw = self._call_gemini_with_retry(prompt)
        if not raw:
            return self._fallback_score()

        score = self._parse_and_validate(raw)
        if score is None:
            return self._fallback_score()

        score["keyword_hits"] = keyword_hits
        score["aggregate"] = self._aggregate(score)
        return score

    def score_session(
        self,
        questions: List[Dict],
        transcripts: Dict[str, Union[Dict, str]],
    ) -> Dict:
        """Score every question in a session and return per-question + overall results."""
        results: List[Dict] = []
        for i, q in enumerate(questions):
            qid = f"q{i+1}"
            transcript = transcripts.get(qid, transcripts.get(str(i)))
            if transcript is None:
                logger.warning(
                    "No transcript for %s — expected key '%s' in transcripts dict", qid, qid
                )
            logger.info("Scoring %s (%s)...", qid, q.get("skill_tested", "?"))
            score = self.score_answer(q, transcript)
            score["question_id"] = qid
            score["question"] = self._question_text(q)
            score["skill_tested"] = q.get("skill_tested", "")
            results.append(score)

        return {
            "per_question": results,
            "overall": self._session_aggregate(results),
        }

    # ─────────────────────────────────────────────────────────────────
    # TRANSCRIPT HANDLING
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _unpack_transcript(transcript: Optional[Union[Dict, str]]) -> Tuple[str, str]:
        """Returns (text, status) where status ∈ {ok, no_answer, silence, transcription_failed}."""
        if transcript is None:
            return "", "no_answer"
        if isinstance(transcript, str):
            text = transcript.strip()
            if not text:
                return "", "no_answer"
            if len(text) < MIN_TRANSCRIPT_CHARS:
                return text, "silence"
            return text, "ok"

        if transcript.get("error"):
            return "", "transcription_failed"
        text = (transcript.get("text") or "").strip()
        if not text:
            return "", "no_answer"
        if len(text) < MIN_TRANSCRIPT_CHARS:
            return text, "silence"
        return text, "ok"

    # ─────────────────────────────────────────────────────────────────
    # PROMPTING
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _question_text(q: Dict) -> str:
        return q.get("question") or q.get("question_text", "")

    def _build_scoring_prompt(self, question: Dict, transcript: str, keyword_hits: List[str]) -> str:
        keywords = question.get("expected_keywords", [])
        return f"""
You are an expert technical interviewer. Score this interview answer.

QUESTION:
{self._question_text(question)}

SKILL TESTED: {question.get('skill_tested', 'general')}
DIFFICULTY: {question.get('difficulty', 'medium')}
EXPECTED KEYWORDS: {', '.join(keywords) if keywords else 'none'}
KEYWORDS THE CANDIDATE ACTUALLY USED: {', '.join(keyword_hits) if keyword_hits else 'none'}

CANDIDATE'S ANSWER (transcribed from audio):
{transcript}

Score 0-10 on each dimension:
1. technical_accuracy: correctness of technical content
2. depth: depth/comprehensiveness
3. relevance: how well it addresses the question
4. clarity: communication quality
5. ai_likeness: 0=clearly human, 10=clearly AI-generated

Also include:
- red_flags: list (empty if none) — e.g. "overly generic", "buzzword salad", "factually incorrect", "off topic", "plagiarized"
- feedback: 2-3 sentences of constructive feedback

Return ONLY valid JSON, no markdown:
{{
  "technical_accuracy": 7,
  "depth": 6,
  "relevance": 8,
  "clarity": 7,
  "ai_likeness": 3,
  "red_flags": [],
  "feedback": "..."
}}

Do NOT include explanations, markdown fences, or any text outside the JSON object.
"""

    def _call_gemini_with_retry(self, prompt: str) -> Optional[str]:
        for attempt in range(2):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                if attempt == 1:
                    logger.error("Gemini API failed after retry: %s", e)
                    return None
                logger.warning("Gemini API call failed, retrying...")
                time.sleep(1)
        return None

    # ─────────────────────────────────────────────────────────────────
    # PARSING & VALIDATION
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_and_validate(raw: str) -> Optional[Dict]:
        """Extract JSON object and verify required numeric fields are sane."""
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                logger.warning("No JSON object in scoring response")
                return None
            score = json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            logger.warning("Score JSON parse failed: %s", e)
            return None

        for dim in SCORE_DIMENSIONS:
            val = score.get(dim)
            if not isinstance(val, (int, float)):
                logger.warning("Missing/invalid '%s' in score response", dim)
                return None
            score[dim] = max(0, min(10, float(val)))

        score.setdefault("red_flags", [])
        score.setdefault("feedback", "")
        if not isinstance(score["red_flags"], list):
            score["red_flags"] = [str(score["red_flags"])]
        return score

    # ─────────────────────────────────────────────────────────────────
    # KEYWORD OVERLAP (deterministic baseline)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _keyword_overlap(question: Dict, transcript: str) -> List[str]:
        keywords = question.get("expected_keywords", []) or []
        # Strip whitespace so "rest api" matches "REST  API" / "rest-api"-style variants
        text_norm = transcript.lower().replace(" ", "")
        return [kw for kw in keywords if kw.lower().replace(" ", "") in text_norm]

    # ─────────────────────────────────────────────────────────────────
    # AGGREGATION
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(score: Dict) -> float:
        """Weighted score 0-10. ai_likeness applied as a penalty."""
        positive = sum(score[dim] * w for dim, w in AGGREGATE_WEIGHTS.items())
        ai_penalty = (score["ai_likeness"] / 10.0) * 2.0  # max 2 point penalty
        return round(max(0.0, positive - ai_penalty), 2)

    @staticmethod
    def _session_aggregate(results: List[Dict]) -> Dict:
        scored = [r for r in results if "aggregate" in r]
        if not scored:
            return {"average": 0.0, "scored_count": 0, "total": len(results), "red_flags": []}

        avg = round(sum(r["aggregate"] for r in scored) / len(scored), 2)
        all_flags = [f for r in scored for f in r.get("red_flags", [])]
        return {
            "average": avg,
            "scored_count": len(scored),
            "total": len(results),
            "red_flags": sorted(set(all_flags)),
        }

    # ─────────────────────────────────────────────────────────────────
    # FALLBACK / EDGE-CASE SCORES
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _zero_score(red_flag: str, feedback: str) -> Dict:
        return {
            "technical_accuracy": 0,
            "depth": 0,
            "relevance": 0,
            "clarity": 0,
            "ai_likeness": 0,
            "red_flags": [red_flag],
            "feedback": feedback,
            "keyword_hits": [],
            "aggregate": 0.0,
        }

    def _no_answer_score(self) -> Dict:
        return self._zero_score("no_answer_provided", "No answer was provided for this question.")

    def _silence_score(self) -> Dict:
        return self._zero_score("silence_or_unintelligible", "Audio contained silence or unintelligible speech.")

    def _transcription_failed_score(self) -> Dict:
        return self._zero_score("transcription_failed", "Audio could not be transcribed — manual review needed.")

    @staticmethod
    def _fallback_score() -> Dict:
        return {
            "technical_accuracy": 5,
            "depth": 5,
            "relevance": 5,
            "clarity": 5,
            "ai_likeness": 5,
            "red_flags": ["scoring_failed"],
            "feedback": "Automatic scoring failed. Manual review recommended.",
            "keyword_hits": [],
            "aggregate": 4.0,
        }


def save_scores(session_result: Dict, output_path: str = "interview_scores.json") -> None:
    """Persist session scoring output to disk."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session_result, f, indent=2, ensure_ascii=False)
    logger.info("Saved scores to %s", output_path)
