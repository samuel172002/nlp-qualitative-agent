"""
NLP Pipeline — orchestrates all 7 stages of the qualitative coding process
without any LLM calls.
"""
from __future__ import annotations
import os
import time
from pathlib import Path

from agent.file_loader import FileLoader
from agent.knowledge_graph import KnowledgeGraphBuilder
from agent.visualizer import Visualizer
from agent.pdf_exporter import PDFExporter
from nlp_first_cycle.engine import FirstCycleEngine
from nlp_second_cycle.engine import SecondCycleEngine
from shared.models import FirstCycleResult, SecondCycleResult


class NLPPipeline:
    """
    End-to-end qualitative coding pipeline using local NLP only.

    Usage::

        pipeline = NLPPipeline()
        result = pipeline.run(
            file_paths=["interview1.txt", "interview2.pdf"],
            research_questions=["How do participants cope?"],
            output_dir="output",
        )
    """

    def __init__(self):
        self._progress_callbacks: list = []

    def add_progress_callback(self, fn) -> None:
        """Register a callback(stage: int, total_stages: int, message: str)."""
        self._progress_callbacks.append(fn)

    def _notify(self, stage: int, total: int, message: str) -> None:
        for fn in self._progress_callbacks:
            try:
                fn(stage, total, message)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        file_paths: list[str],
        research_questions: list[str] | None = None,
        output_dir: str = "output",
        n_clusters: int | None = None,
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        total_stages = 7
        start = time.time()

        # ── Stage 1: Load & segment ─────────────────────────────────────
        self._notify(1, total_stages, "Loading and segmenting files…")
        loader = FileLoader()
        segments = loader.load(file_paths)
        if not segments:
            raise ValueError("No text could be extracted from the provided files.")

        # ── Stage 2: First Cycle coding ─────────────────────────────────
        self._notify(2, total_stages, "Running first-cycle NLP coding (9 methods)…")

        def fc_progress(done, total_segs):
            msg = f"First cycle: segment {done}/{total_segs}"
            self._notify(2, total_stages, msg)

        fc_engine = FirstCycleEngine(research_questions=research_questions or [])
        fc_result: FirstCycleResult = fc_engine.run(segments, progress_callback=fc_progress)

        # ── Stage 3: Second Cycle coding ────────────────────────────────
        self._notify(3, total_stages, "Running second-cycle coding (pattern → theory)…")

        # Reuse the already-loaded spaCy model from first cycle resources
        from nlp_first_cycle.coders import NLPResources
        nlp = NLPResources().nlp if NLPResources()._loaded else None

        def sc_progress(done, total_steps, label):
            self._notify(3, total_stages, f"Second cycle [{done}/{total_steps}]: {label}")

        sc_engine = SecondCycleEngine(n_clusters=n_clusters, spacy_nlp=nlp)
        sc_result: SecondCycleResult = sc_engine.run(fc_result, progress_callback=sc_progress)

        # ── Stage 4: Build knowledge graph ──────────────────────────────
        self._notify(4, total_stages, "Building knowledge graph…")
        graph = KnowledgeGraphBuilder().build(fc_result, sc_result)

        # ── Stage 5: Export visualizations ──────────────────────────────
        self._notify(5, total_stages, "Exporting visualizations…")
        Visualizer().export(graph, output_dir, fc=fc_result, sc=sc_result)

        # ── Stage 6: Build summary ───────────────────────────────────────
        self._notify(6, total_stages, "Building summary…")
        summary = self._build_summary(file_paths, segments, fc_result, sc_result,
                                      elapsed=time.time() - start)

        # ── Stage 7: Export PDF ──────────────────────────────────────────
        self._notify(7, total_stages, "Exporting PDF report…")
        pdf_path = PDFExporter().export(summary, fc_result, sc_result, output_dir)
        summary["pdf_path"] = pdf_path

        self._notify(7, total_stages, "Done.")
        return summary

    # ------------------------------------------------------------------ #
    # Summary builder
    # ------------------------------------------------------------------ #
    def _build_summary(
        self,
        file_paths: list[str],
        segments,
        fc: FirstCycleResult,
        sc: SecondCycleResult,
        elapsed: float,
    ) -> dict:
        top_codes = sorted(
            fc.code_frequencies.items(), key=lambda x: x[1], reverse=True
        )[:15]

        return {
            "file_count": len(file_paths),
            "files": [Path(p).name for p in file_paths],
            "total_segments": len(segments),
            "total_codes": sum(fc.code_frequencies.values()),
            "unique_code_labels": len(fc.all_codes),
            "top_codes": top_codes,
            "category_count": len(sc.categories),
            "categories": [
                {
                    "name": c.name,
                    "frequency": c.frequency,
                    "code_count": len(c.codes),
                }
                for c in sc.categories
            ],
            "relationship_count": len(sc.axial_relationships),
            "theme_count": len(sc.themes),
            "core_category": (
                sc.core_category.name if sc.core_category else None
            ),
            "theoretical_statement": (
                sc.core_category.theoretical_statement
                if sc.core_category else ""
            ),
            "elapsed_seconds": round(elapsed, 1),
        }
