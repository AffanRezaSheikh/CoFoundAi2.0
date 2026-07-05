"""
Transcribe interview audio recordings to text using OpenAI Whisper.

Designed to run after an interview session: takes per-question audio files,
returns clean transcripts plus segment metadata for downstream scoring.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
MIN_AUDIO_BYTES = 1024  # anything smaller is almost certainly a corrupt/empty recording


class AudioTranscriber:
    """
    Lazy-loaded Whisper wrapper for interview transcription.

    Args:
        model_size: tiny | base | small | medium | large. "base" is a good
                    interview default (fast on CPU, accurate enough for grading).
        language:   ISO code (e.g. "en"). None = auto-detect.
        device:     "cuda" | "cpu" | None (auto).
        cache_dir:  if set, transcripts are cached as JSON next to source audio
                    so re-runs skip already-transcribed files.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = "en",
        device: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None  # lazy-loaded

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            import whisper
        except ImportError as e:
            raise ImportError(
                "openai-whisper is not installed.\n"
                "  pip install openai-whisper\n"
                "  (also requires ffmpeg: https://ffmpeg.org)"
            ) from e

        logger.info("Loading Whisper model '%s'...", self.model_size)
        self._model = whisper.load_model(self.model_size, device=self.device)
        logger.info("Whisper model ready.")
        return self._model

    def _validate_audio(self, audio_path: Path) -> Optional[str]:
        """Return error string if invalid, else None."""
        if not audio_path.exists():
            return f"file not found: {audio_path}"
        if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
            return f"unsupported format '{audio_path.suffix}'"
        if audio_path.stat().st_size < MIN_AUDIO_BYTES:
            return f"audio too small ({audio_path.stat().st_size} bytes) — likely empty"
        return None

    def _cache_path(self, audio_path: Path) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Hash the resolved path so two files with the same stem don't collide
        path_hash = hashlib.sha1(str(audio_path.resolve()).encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"{audio_path.stem}_{path_hash}.json"

    def transcribe(self, audio_path: Union[str, Path]) -> Dict:
        """
        Transcribe a single audio file.

        Returns:
            {
                "text": str,                # full transcript (empty string if silence)
                "language": str,            # detected/used language
                "duration": float,          # seconds
                "segments": List[Dict],     # whisper segments with timestamps
                "error": Optional[str],     # set if transcription failed
            }
        """
        audio_path = Path(audio_path)

        err = self._validate_audio(audio_path)
        if err:
            logger.warning("Skipping %s: %s", audio_path.name, err)
            return self._empty_result(error=err)

        cache_file = self._cache_path(audio_path)
        if cache_file and cache_file.exists():
            logger.info("Cache hit for %s", audio_path.name)
            return json.loads(cache_file.read_text(encoding="utf-8"))

        try:
            model = self._load_model()
            result = model.transcribe(
                str(audio_path),
                language=self.language,
                fp16=False,  # safer on CPU; Whisper warns otherwise
                verbose=False,
            )
        except Exception as e:
            logger.exception("Transcription failed for %s", audio_path)
            return self._empty_result(error=str(e))

        segments: List[Dict] = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]
        # Collapse newlines / runs of whitespace for cleaner downstream matching
        clean_text = " ".join((result.get("text") or "").split())
        out = {
            "text": clean_text,
            "language": result.get("language", self.language or "unknown"),
            "duration": segments[-1]["end"] if segments else 0.0,
            "segments": segments,
            "error": None,
        }

        if cache_file:
            cache_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        return out

    def transcribe_session(
        self,
        audio_files: Dict[str, Union[str, Path]],
    ) -> Dict[str, Dict]:
        """
        Transcribe all answers in an interview session.

        Args:
            audio_files: {question_id: audio_path}

        Returns:
            {question_id: transcribe() result dict}
        """
        transcripts: Dict[str, Dict] = {}
        total = len(audio_files)

        for i, (qid, path) in enumerate(audio_files.items(), 1):
            logger.info("[%d/%d] Transcribing %s...", i, total, qid)
            result = self.transcribe(path)
            transcripts[qid] = result

            if result["error"]:
                logger.warning("  %s: %s", qid, result["error"])
            elif len(result["text"]) < 5:
                # Whisper sometimes returns "" or filler like "you" on silence
                logger.warning("  %s: empty/garbage transcript (silence?)", qid)
            else:
                logger.info("  %s: %d chars, %.1fs", qid, len(result["text"]), result["duration"])

        return transcripts

    @staticmethod
    def _empty_result(error: Optional[str] = None) -> Dict:
        return {
            "text": "",
            "language": "unknown",
            "duration": 0.0,
            "segments": [],
            "error": error,
        }


def save_transcripts(transcripts: Dict[str, Dict], output_path: Union[str, Path]) -> None:
    """Persist a session transcript bundle to JSON."""
    Path(output_path).write_text(
        json.dumps(transcripts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d transcripts to %s", len(transcripts), output_path)
