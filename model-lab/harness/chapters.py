"""
Chapters Core Logic - Semantic Segmentation & Timeline Indexing.

Implements algorithm to split long audio into chapters based on topic shifts:
1. Sliding window embeddings (SentenceTransformers)
2. Cosine similarity profile
3. Peak detection for boundaries
4. Metadata enrichment (LLM)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json

from harness.alignment import AlignedSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Exposed at module level for unit tests to patch without importing heavy deps.
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

@dataclass
class Chapter:
    index: int
    start: float
    end: float
    title: str = "New Chapter"
    summary: str = ""
    speakers: List[str] = field(default_factory=list)
    confidence: float = 1.0
    evidence_segments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": f"ch_{self.index:02d}",
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "title": self.title,
            "summary": self.summary,
            "speakers": self.speakers,
            "confidence": self.confidence,
            "evidence": self.evidence_segments
        }

class SemanticSegmenter:
    """Segments transcript into chapters using semantic embeddings."""
    
    DEFAULT_MODEL = 'all-MiniLM-L6-v2'
    WINDOW_SIZE_SEC = 60.0
    STRIDE_SEC = 15.0
    MIN_CHAPTER_SEC = 45.0
    
    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Optional[Path] = None):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed. Install with: uv add sentence-transformers")

        logger.info(f"Loading embedding model: {model_name}")
        if cache_dir:
            logger.info(f"Using cache directory: {cache_dir}")
            
        self.model_name = model_name
        self.cache_dir = str(cache_dir) if cache_dir else None
        
        # Load model with explicit cache
        self.model = SentenceTransformer(model_name, cache_folder=self.cache_dir)
        logger.info("Model loaded.")

    def segment(self, segments: List[AlignedSegment], duration: float) -> Tuple[List[Chapter], Dict[str, Any]]:
        """
        Segment aligned text into chapters.
        
        Args:
            segments: List of AlignedSegment (from alignment)
            duration: Total duration of audio in seconds
            
        Returns:
            Tuple[List[Chapter], Dict[str, Any]]: Chapters and metadata/config
        """
        try:
             from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
             raise ImportError("scikit-learn not installed. Install with: uv add scikit-learn")
        
        config = {
            "model": self.model_name,
            "window_size": self.WINDOW_SIZE_SEC,
            "stride": self.STRIDE_SEC,
            "min_chapter_duration": self.MIN_CHAPTER_SEC,  
            "threshold_strategy": "mean_minus_half_std"
        }
        
        # 1. Gates
        if duration < 120.0:
            logger.warning(f"Audio too short for segmentation ({duration:.1f}s < 120s). Returning single chapter.")
            return [self._create_chapter(0, 0.0, duration, segments)], {**config, "gating": "short_duration"}

        if not segments:
            return [], config
            
        # Check density (approx words/min) -> maybe later if needed.
        
        # 2. Create Sliding Windows
        windows = self._create_windows(segments, duration)
        if not windows:
            return [self._create_chapter(0, 0.0, duration, segments)], {**config, "gating": "no_windows"}
            
        # 3. Compute Embeddings
        texts = [w['text'] for w in windows]
        embeddings = self.model.encode(texts)
        
        # 4. Compute Similarity Profile
        if len(embeddings) < 2:
            return [self._create_chapter(0, 0.0, duration, segments)], config
            
        sims = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1), 
                embeddings[i+1].reshape(1, -1)
            )[0][0]
            sims.append(sim)
            
        # 5. Find Boundaries
        boundaries = [0.0] # Start
        last_boundary = 0.0
        
        sim_arr = np.array(sims)
        threshold = np.mean(sim_arr) - (0.5 * np.std(sim_arr))
        config["threshold_value"] = float(threshold)
        
        candidate_indices = []
        for i in range(1, len(sims) - 1):
            if sims[i] < sims[i-1] and sims[i] < sims[i+1]:
                if sims[i] < threshold:
                    candidate_indices.append(i)
        
        candidate_times = []
        for idx in candidate_indices:
            t = (idx * self.STRIDE_SEC) + (self.WINDOW_SIZE_SEC / 2)
            candidate_times.append(t)
            
        for t in candidate_times:
            if t - last_boundary >= self.MIN_CHAPTER_SEC:
                if duration - t >= self.MIN_CHAPTER_SEC:
                    boundaries.append(t)
                    last_boundary = t
                    
        boundaries.append(duration)
        config["boundary_count"] = len(boundaries)
        
        # 6. Construct Chapters
        chapters = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            
            # Gather segments (strict time inclusion)
            chap_segsi = []
            speakers = set()
            for seg in segments:
                mid = (seg.start_s + seg.end_s) / 2
                if start <= mid < end:
                    chap_segsi.append(seg)
                    if seg.speaker_id and seg.speaker_id != "unknown":
                        speakers.add(seg.speaker_id)
            
            # Create object
            chapters.append(Chapter(
                index=i+1,
                start=start,
                end=end,
                speakers=sorted(list(speakers)),
                evidence_segments=[{
                    "text": s.text, 
                    "start": s.start_s, 
                    "end": s.end_s
                } for s in chap_segsi]
            ))
            
        return chapters, config

    def _create_chapter(self, index, start, end, segments):
        """Helper for single chapter fallback."""
        speakers = set(s.speaker_id for s in segments if s.speaker_id != "unknown")
        return Chapter(
            index=index+1,
            start=start,
            end=end,
            speakers=sorted(list(speakers)),
            evidence_segments=[{
                "text": s.text, 
                "start": s.start_s, 
                "end": s.end_s
            } for s in segments]
        )

    def _create_windows(self, segments, duration) -> List[Dict]:
        """Generate text windows."""
        windows = []
        curr_time = 0.0
        
        while curr_time + self.WINDOW_SIZE_SEC <= duration + self.STRIDE_SEC: # inclusive-ish
            end_time = curr_time + self.WINDOW_SIZE_SEC
            
            # Collect text
            text_parts = []
            for seg in segments:
                # Check overlap
                mid = (seg.start_s + seg.end_s) / 2
                if curr_time <= mid < end_time:
                    text_parts.append(seg.text)
            
            combined_text = " ".join(text_parts).strip()
            if combined_text:
                windows.append({
                    "start": curr_time,
                    "end": end_time,
                    "text": combined_text
                })
                
            curr_time += self.STRIDE_SEC
            
        return windows

# --- Orchestration Logic ---
from harness.alignment import load_alignment
from harness.llm_provider import get_llm_completion
from harness.nlp_schema import compute_file_hash

CHAPTER_PROMPT = """You are an editor creating a Table of Contents for a meeting transcript.
Analyze the following section of text and provide:
1. A concise Title (max 10 words).
2. A short Summary (1-2 sentences).

TEXT:
{text}

Respond in JSON:
{{
  "title": "...",
  "summary": "..."
}}
"""

def enrich_chapter(chapter: Chapter, context: str):
    """Enrich chapter with Title/Summary from LLM."""
    prompt = CHAPTER_PROMPT.format(text=context[:5000]) # Safe limit
    
    result = get_llm_completion(prompt)
    if result.success:
        try:
            # Clean possible markdown code blocks
            clean_text = result.text.strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_text)
            chapter.title = data.get("title", chapter.title)
            chapter.summary = data.get("summary", chapter.summary)
        except json.JSONDecodeError:
            logging.getLogger(__name__).warning(f"Failed to parse LLM response for chapter {chapter.index}")
    else:
        logging.getLogger(__name__).warning(f"LLM failure for chapter {chapter.index}: {result.error_message}")

def run_chapters(
    alignment_path: Path, 
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    force: bool = False
) -> Path:
    """
    Run chapters extraction pipeline.
    
    Args:
        alignment_path: Path to alignment artifact.
        output_dir: Directory to save artifact.
        cache_dir: Cache directory for embedding models.
        force: Overwrite existing.
        
    Returns:
        Path to chapters artifact.
    """
    alignment_path = alignment_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    if not alignment_path.exists():
        raise FileNotFoundError(f"Alignment not found: {alignment_path}")
        
    aligned_transcript = load_alignment(alignment_path)
    
    # 2. Segment
    if cache_dir:
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        
    segmenter = SemanticSegmenter(cache_dir=cache_dir)
    
    if aligned_transcript.segments:
        duration = aligned_transcript.segments[-1].end_s + 2.0
    else:
        # Handle empty
        duration = 0.0
        
    chapters, config = segmenter.segment(aligned_transcript.segments, duration)
    
    # 3. Enrich (LLM)
    for chap in chapters:
        texts = [s["text"] for s in chap.evidence_segments]
        full_text = " ".join(texts)
        if not full_text.strip():
            continue
            
        enrich_chapter(chap, full_text)
        
    # 4. Save Artifact
    input_hash = compute_file_hash(alignment_path)
    
    output_data = {
        "chapters": [c.to_dict() for c in chapters],
        "stats": {
            "count": len(chapters),
            "avg_duration": sum(c.end - c.start for c in chapters) / len(chapters) if chapters else 0
        },
        "config": config,
        "inputs": {
            "parent_artifact_path": str(alignment_path),
            "parent_artifact_hash": input_hash
        }
    }
    
    artifact_name = f"chapters_{alignment_path.stem.replace('alignment_', '')}.json"
    output_path = output_dir / artifact_name
    
    # Check if chapters_ prefix is duplicated? alignment_mock -> chapters_mock.
    # If file is alignment_mock.json -> stem is alignment_mock.
    # chapters_alignment_mock.json is what we had.
    # The previous code: f"chapters_{alignment_path.stem}.json"
    # If alignment_path.stem is 'alignment_foo', result is 'chapters_alignment_foo.json'.
    # In integration test we expected 'artifacts/nlp_chapters_chapters_alignment_mock.json' (from export bundle naming).
    # But run_chapters produced 'chapters_alignment_mock.json'.
    # I'll stick to f"chapters_{alignment_path.stem}.json" to match previous behavior.
    
    output_path = output_dir / f"chapters_{alignment_path.stem}.json"
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    return output_path

