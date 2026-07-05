"""
AI Hiring System - Interview Pipeline

Per-candidate stages:
    1. CV Parsing       (cv_extractor.py)
    2. Question Gen     (generate_question.py)
    3. Transcription    (transcriptions_pro.py)
    4. Answer Scoring   (score.py)
    5. Aggregation      (aggregrate.py)
    6. Report assembly  (here)

Cohort stage (run separately over a list of candidate reports):
    - Bias Audit        (fairness.py) — see run_bias_audit_on_cohort()
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from HR.cv_extractor import CVExtractor, pdf_to_text
from HR.generate_question import QuestionGenerator
from HR.transcriptions_pro import AudioTranscriber
from HR.score import AnswerScorer
from aggregrate import aggregate_candidate_scores
from fairness import BiasAuditor

load_dotenv()
logger = logging.getLogger(__name__)


class StageError(Exception):
    """Raised when a pipeline stage fails irrecoverably."""


class InterviewPipeline:
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        # Lazy: question generator, transcriber, scorer load on first use
        self.question_generator: Optional[QuestionGenerator] = None
        self.transcriber: Optional[AudioTranscriber] = None
        self.scorer: Optional[AnswerScorer] = None
        self.auditor = BiasAuditor()

    # ─────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────

    def process_candidate(
            self,
            cv_path: str,
            audio_files: Dict[str, str],
            recruiter_config: Dict,
            candidate_metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Run the full per-candidate pipeline. Each stage is isolated:
        a stage failure is recorded under `stages_failed` and the pipeline
        continues with degraded data instead of raising.
        """
        logger.info("=" * 60)
        logger.info("Processing candidate: %s", cv_path)
        logger.info("=" * 60)

        report: Dict = {
            "candidate_id": Path(cv_path).stem,
            "metadata": candidate_metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stages_failed": [],
        }

        # Stage 1 — CV parsing
        cv_json = self._stage("cv_parsing", report, lambda: self._parse_cv(cv_path)) or {}
        report["cv_data"] = cv_json
        logger.info("[1/6] CV parsed: %s, %d skills",
                    cv_json.get("name", "Unknown"),
                    len(cv_json.get("skills", []) or []))

        # Stage 2 — Question generation
        questions = self._stage(
            "question_generation",
            report,
            lambda: self._generate_questions(cv_json, recruiter_config),
        ) or []
        report["questions"] = questions
        logger.info("[2/6] Generated %d questions", len(questions))

        # Stage 3 — Transcription
        transcripts = self._stage(
            "transcription",
            report,
            lambda: self._transcribe(audio_files),
        ) or {}
        report["transcripts"] = transcripts
        ok = sum(1 for t in transcripts.values() if isinstance(t, dict) and t.get("text"))
        logger.info("[3/6] Transcribed %d / %d audio files", ok, len(transcripts))

        # Stage 4 — Scoring
        scoring = self._stage(
            "scoring",
            report,
            lambda: self._score(questions, transcripts),
        ) or {"per_question": [], "overall": {}}
        report["scoring"] = scoring
        logger.info("[4/6] Scored %d answers", len(scoring.get("per_question", [])))

        # Stage 5 — Aggregation (passes questions for difficulty/category breakdowns)
        aggregated = self._stage(
            "aggregation",
            report,
            lambda: aggregate_candidate_scores(scoring.get("per_question", []), questions=questions),
        ) or {}
        report["aggregated_scores"] = aggregated
        logger.info(
            "[5/6] Overall %.2f / 10 — %s",
            aggregated.get("overall_score", 0.0),
            aggregated.get("recommendation", "n/a"),
        )

        # Stage 6 — Report finalization
        logger.info("[6/6] Report assembled (failed stages: %s)",
                    report["stages_failed"] or "none")
        return report

    # ─────────────────────────────────────────────────────────────────
    # STAGE WRAPPER
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _stage(name: str, report: Dict, fn):
        """Run a stage; on failure, log + tag the report and return None."""
        try:
            return fn()
        except Exception as e:
            logger.exception("Stage '%s' failed: %s", name, e)
            report["stages_failed"].append({"stage": name, "error": str(e)})
            return None

    # ─────────────────────────────────────────────────────────────────
    # STAGE IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_cv(cv_path: str) -> Dict:
        text = pdf_to_text(cv_path)
        return CVExtractor(text).extract_all()

    def _generate_questions(self, cv_json: Dict, config: Dict) -> List[Dict]:
        if self.question_generator is None:
            self.question_generator = QuestionGenerator(self.gemini_api_key)
        return self.question_generator.generate_questions(cv_json, config)

    def _transcribe(self, audio_files: Dict[str, str]) -> Dict[str, Dict]:
        if self.transcriber is None:
            self.transcriber = AudioTranscriber(model_size="base")
        return self.transcriber.transcribe_session(audio_files)

    def _score(self, questions: List[Dict], transcripts: Dict[str, Dict]) -> Dict:
        if self.scorer is None:
            self.scorer = AnswerScorer(self.gemini_api_key)
        return self.scorer.score_session(questions, transcripts)


# ────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────

def save_report(report: Dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", output_path)


def run_bias_audit_on_cohort(
        reports: List[Dict],
        protected_attr: str = "gender",
) -> Dict:
    """Cohort-level bias audit over a list of per-candidate reports."""
    records = []
    for report in reports:
        agg = report.get("aggregated_scores", {})
        records.append({
            "candidate_id": report.get("candidate_id"),
            "overall_score": agg.get("overall_score"),
            "recommendation": agg.get("recommendation"),
            protected_attr: report.get("metadata", {}).get(protected_attr, "unknown"),
        })

    df = pd.DataFrame(records)
    return BiasAuditor().audit_cohort(df, protected_attr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Interview Pipeline module — import InterviewPipeline to use.")
