"""
Second Cycle Engine — runs all 4 NLP coders on a FirstCycleResult
and assembles a SecondCycleResult.
"""
from __future__ import annotations

from shared.models import FirstCycleResult, SecondCycleResult
from nlp_second_cycle.coders import (
    PatternCoder,
    FocusedCoder,
    AxialCoder,
    TheoreticalCoder,
)


class SecondCycleEngine:
    def __init__(self, n_clusters: int | None = None, spacy_nlp=None):
        """
        n_clusters: override number of clusters (default: auto)
        spacy_nlp: pass in loaded spaCy model for richer property extraction;
                   if None, property extraction is skipped.
        """
        self._n_clusters = n_clusters
        self._nlp = spacy_nlp

    def run(self, fc: FirstCycleResult,
            progress_callback=None) -> SecondCycleResult:

        def _step(n, label):
            if progress_callback:
                progress_callback(n, 4, label)

        # Stage 3a: Pattern coding — group codes into meta-patterns
        _step(1, "Pattern coding…")
        pattern_coder = PatternCoder(n_clusters=self._n_clusters)
        pattern_codes = pattern_coder.code(fc)

        # Stage 3b: Focused coding — derive categories from patterns
        _step(2, "Focused coding…")
        focused_coder = FocusedCoder(nlp=self._nlp)
        categories = focused_coder.code(fc, pattern_codes)

        # Stage 3c: Axial coding — find relationships between categories
        _step(3, "Axial coding…")
        axial_coder = AxialCoder()
        axial_relationships = axial_coder.code(fc, categories)

        # Stage 3d: Theoretical coding — core category + themes
        _step(4, "Theoretical coding…")
        theoretical_coder = TheoreticalCoder()
        core_category, themes = theoretical_coder.code(fc, categories)

        return SecondCycleResult(
            categories=categories,
            axial_relationships=axial_relationships,
            themes=themes,
            core_category=core_category,
            pattern_codes=pattern_codes,
        )
