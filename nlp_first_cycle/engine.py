"""
First Cycle Engine — runs all 9 NLP coders on every segment,
deduplicates, and assembles a FirstCycleResult.
"""
from __future__ import annotations
from collections import defaultdict

from shared.models import (
    Code, CodedSegment, FirstCycleResult, TextSegment,
)
from nlp_first_cycle.coders import (
    NLPResources,
    DescriptiveCoder,
    InVivoCoder,
    ProcessCoder,
    InitialCoder,
    StructuralCoder,
    EmotionCoder,
    ValuesCoder,
    VersusCoder,
    EvaluationCoder,
)


class FirstCycleEngine:
    def __init__(self, research_questions: list[str] | None = None):
        self.research_questions = research_questions or []

    def run(self, segments: list[TextSegment],
            progress_callback=None) -> FirstCycleResult:
        # Load models once
        resources = NLPResources()
        resources.load(self.research_questions)

        # Build TF-IDF corpus from all segment texts for DescriptiveCoder
        corpus_texts = [s.text for s in segments]
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                                  max_features=500)
            mat = vec.fit_transform(corpus_texts)
            corpus_scores = dict(zip(
                vec.get_feature_names_out(),
                mat.toarray().mean(axis=0).tolist()
            ))
        except Exception:
            corpus_scores = {}

        # Instantiate all coders
        coders = [
            DescriptiveCoder(resources, corpus_texts),
            InVivoCoder(resources),
            ProcessCoder(resources),
            InitialCoder(resources),
            StructuralCoder(resources),
            EmotionCoder(resources),
            ValuesCoder(resources),
            VersusCoder(resources),
            EvaluationCoder(resources),
        ]
        # Patch corpus scores into DescriptiveCoder
        coders[0].corpus_scores = corpus_scores

        coded_segments: list[CodedSegment] = []
        code_frequencies: dict[str, int] = defaultdict(int)
        all_codes: dict[str, list[Code]] = defaultdict(list)

        total = len(segments)
        for i, segment in enumerate(segments):
            all_segment_codes: list[Code] = []
            seen_labels: set[str] = set()

            for coder in coders:
                try:
                    new_codes = coder.code(segment)
                except Exception:
                    new_codes = []

                for code in new_codes:
                    # Deduplicate by label within the same segment
                    if code.label not in seen_labels and len(code.label) >= 3:
                        seen_labels.add(code.label)
                        all_segment_codes.append(code)
                        code_frequencies[code.label] += 1
                        all_codes[code.label].append(code)

            coded_segments.append(CodedSegment(segment=segment,
                                               codes=all_segment_codes))

            if progress_callback:
                progress_callback(i + 1, total)

        return FirstCycleResult(
            coded_segments=coded_segments,
            code_frequencies=dict(code_frequencies),
            all_codes=dict(all_codes),
        )
