"""
Second-cycle NLP coders — 4 methods from Saldaña's Coding Manual.
Input: FirstCycleResult
Output: pattern_codes, categories, axial_relationships, core_category, themes
"""
from __future__ import annotations
import re
from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from shared.models import (
    AxialRelationship,
    Category,
    Code,
    CoreCategory,
    FirstCycleResult,
    Theme,
)

# Minimum segment co-occurrences to create an axial relationship
_CO_OCCUR_THRESHOLD = 2

# Minimum code frequency to count as "focused" / salient
_MIN_FREQ = 2

# Causal / constraint verb lemmas for relationship type heuristic
_CAUSAL_VERBS = {"cause", "lead", "result", "produce", "create",
                 "generate", "trigger", "drive", "enable", "facilitate"}
_CONSTRAINT_VERBS = {"limit", "prevent", "block", "constrain", "restrict",
                     "inhibit", "hinder", "impede", "stop", "reduce"}

# Hedging words that mark latent (inferential) themes
_HEDGING = {"suggest", "suggest", "imply", "implication", "may", "might",
            "could", "seem", "appear", "indicate", "reflect", "underlying"}


# ---------------------------------------------------------------------------
# Helper: build one "document" per code from its excerpts
# ---------------------------------------------------------------------------

def _code_documents(fc: FirstCycleResult) -> tuple[list[str], list[str]]:
    """Returns (labels_list, docs_list) — one doc per unique code label."""
    labels = []
    docs = []
    for label, codes in fc.all_codes.items():
        labels.append(label)
        doc = " ".join(c.excerpt for c in codes if c.excerpt)
        if not doc:
            doc = label.lower()
        docs.append(doc)
    return labels, docs


def _top_terms(tfidf_matrix, feature_names: list[str],
               cluster_mask: np.ndarray, top_n: int = 5) -> list[str]:
    """Return top TF-IDF terms for a cluster."""
    if not cluster_mask.any():
        return []
    cluster_vecs = tfidf_matrix[cluster_mask]
    mean_vec = cluster_vecs.mean(axis=0)
    if hasattr(mean_vec, "A1"):        # sparse matrix row
        mean_vec = mean_vec.A1
    top_indices = mean_vec.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_indices if mean_vec[i] > 0]


def _top_pos_words(codes: list[Code], nlp, pos_tags: set[str],
                   top_n: int = 5) -> list[str]:
    """Extract top words with specified POS tags from a list of codes."""
    from collections import Counter
    counts: Counter = Counter()
    for code in codes:
        text = code.excerpt or ""
        doc = nlp(text[:200])
        for tok in doc:
            if tok.pos_ in pos_tags and not tok.is_stop and tok.is_alpha and len(tok.lemma_) > 2:
                counts[tok.lemma_.lower()] += 1
    return [w for w, _ in counts.most_common(top_n)]


# ---------------------------------------------------------------------------
# 1. Pattern Coder
# ---------------------------------------------------------------------------

class PatternCoder:
    """Groups first-cycle codes into meta-patterns via TF-IDF + clustering."""

    def __init__(self, n_clusters: int | None = None):
        self._n_clusters = n_clusters

    def code(self, fc: FirstCycleResult) -> dict[str, list[str]]:
        labels, docs = _code_documents(fc)
        if len(labels) < 3:
            return {"GENERAL PATTERN": labels}

        n_clusters = self._n_clusters or max(3, min(10, len(labels) // 4))

        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=300, stop_words="english",
                                     ngram_range=(1, 2))
        try:
            X = vectorizer.fit_transform(docs).toarray()
            feature_names = list(vectorizer.get_feature_names_out())
        except ValueError:
            return {"GENERAL PATTERN": labels}

        # Filter out zero-norm rows — cosine affinity cannot handle them
        norms = np.linalg.norm(X, axis=1)
        valid_mask = norms > 0
        X_valid = X[valid_mask]
        labels_valid = [labels[i] for i in range(len(labels)) if valid_mask[i]]
        labels_zero = [labels[i] for i in range(len(labels)) if not valid_mask[i]]

        if len(labels_valid) < 3:
            return {"GENERAL PATTERN": labels}

        # Ensure n_clusters ≤ n_valid_samples
        n_clusters = min(n_clusters, len(labels_valid))

        # Agglomerative clustering with cosine distance
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        cluster_ids = clustering.fit_predict(X_valid)

        # Name each cluster by its top TF-IDF term
        pattern_codes: dict[str, list[str]] = {}
        for cid in range(n_clusters):
            mask = cluster_ids == cid
            member_labels = [labels_valid[i] for i in range(len(labels_valid)) if cluster_ids[i] == cid]
            if not member_labels:
                continue
            top = _top_terms(X_valid, feature_names, mask)
            pattern_name = top[0].upper() if top else f"PATTERN_{cid + 1}"
            # Avoid duplicate pattern names
            base = pattern_name
            suffix = 2
            while pattern_name in pattern_codes:
                pattern_name = f"{base}_{suffix}"
                suffix += 1
            pattern_codes[pattern_name] = member_labels

        # Absorb zero-norm labels into the largest cluster
        if labels_zero and pattern_codes:
            largest = max(pattern_codes, key=lambda k: len(pattern_codes[k]))
            pattern_codes[largest].extend(labels_zero)

        return pattern_codes


# ---------------------------------------------------------------------------
# 2. Focused Coder
# ---------------------------------------------------------------------------

class FocusedCoder:
    """Derives focused categories from pattern clusters, filtered to salient codes."""

    def __init__(self, nlp=None):
        self._nlp = nlp  # optional spaCy model for property extraction

    def code(self, fc: FirstCycleResult,
             pattern_codes: dict[str, list[str]]) -> list[Category]:
        categories: list[Category] = []

        for pattern_name, code_labels in pattern_codes.items():
            # Filter to salient (frequent) codes
            salient = [lbl for lbl in code_labels
                       if fc.code_frequencies.get(lbl, 0) >= _MIN_FREQ]
            if not salient:
                salient = code_labels   # keep all if none pass threshold

            freq = sum(fc.code_frequencies.get(lbl, 1) for lbl in salient)

            # Description = top 3 terms from member code labels
            label_text = " ".join(salient).lower()
            try:
                vec = TfidfVectorizer(max_features=10, stop_words="english")
                vec.fit_transform([label_text])
                desc_terms = list(vec.get_feature_names_out())[:3]
                description = ", ".join(desc_terms) if desc_terms else pattern_name.lower()
            except Exception:
                description = pattern_name.lower()

            # Properties and dimensions from excerpts
            all_member_codes = []
            for lbl in salient:
                all_member_codes.extend(fc.all_codes.get(lbl, []))

            properties: list[str] = []
            dimensions: list[str] = []
            if self._nlp and all_member_codes:
                props = _top_pos_words(all_member_codes, self._nlp, {"ADJ"}, top_n=4)
                properties = props

                # Dimension: polarity range based on +/− labels
                has_pos = any("+" in c.label for c in all_member_codes)
                has_neg = any("-" in c.label for c in all_member_codes)
                if has_pos and has_neg:
                    dimensions = ["positive ↔ negative"]
                elif has_pos:
                    dimensions = ["positive"]
                elif has_neg:
                    dimensions = ["negative"]

            categories.append(Category(
                name=pattern_name,
                description=description,
                codes=salient,
                frequency=freq,
                properties=properties,
                dimensions=dimensions,
            ))

        # Sort by frequency descending
        categories.sort(key=lambda c: c.frequency, reverse=True)
        return categories


# ---------------------------------------------------------------------------
# 3. Axial Coder
# ---------------------------------------------------------------------------

class AxialCoder:
    """Finds causal/relational links between categories via segment co-occurrence."""

    def code(self, fc: FirstCycleResult,
             categories: list[Category]) -> list[AxialRelationship]:
        # Map code label → category name
        code_to_cat: dict[str, str] = {}
        for cat in categories:
            for code_label in cat.codes:
                code_to_cat[code_label] = cat.name

        # Count co-occurrences
        co_occur: dict[tuple[str, str], int] = defaultdict(int)
        for cs in fc.coded_segments:
            cats_in_seg: set[str] = set()
            for code in cs.codes:
                cat = code_to_cat.get(code.label)
                if cat:
                    cats_in_seg.add(cat)
            for a, b in combinations(sorted(cats_in_seg), 2):
                co_occur[(a, b)] += 1

        # Build relationships above threshold
        relationships: list[AxialRelationship] = []
        for (src, tgt), count in co_occur.items():
            if count < _CO_OCCUR_THRESHOLD:
                continue
            rel_type = self._infer_relationship_type(src, tgt, fc, code_to_cat)
            relationships.append(AxialRelationship(
                source_category=src,
                target_category=tgt,
                relationship_type=rel_type,
                description=f"Co-occur in {count} segment(s)",
                conditions=[],
                consequences=[],
            ))

        # Sort by co-occurrence (proxy: description contains highest number)
        relationships.sort(
            key=lambda r: int(re.search(r"\d+", r.description).group()),
            reverse=True,
        )
        return relationships

    def _infer_relationship_type(self, src: str, tgt: str,
                                  fc: FirstCycleResult,
                                  code_to_cat: dict[str, str]) -> str:
        """Heuristic: inspect excerpts of both categories for verb cues."""
        src_codes = [c for lbl, codes in fc.all_codes.items()
                     for c in codes if code_to_cat.get(lbl) == src]
        tgt_codes = [c for lbl, codes in fc.all_codes.items()
                     for c in codes if code_to_cat.get(lbl) == tgt]
        all_excerpts = " ".join(
            c.excerpt.lower() for c in src_codes + tgt_codes if c.excerpt
        )
        words = set(re.findall(r"\b\w+\b", all_excerpts))
        if words & _CAUSAL_VERBS:
            return "causes"
        if words & _CONSTRAINT_VERBS:
            return "constrains"
        return "leads_to"


# ---------------------------------------------------------------------------
# 4. Theoretical Coder
# ---------------------------------------------------------------------------

class TheoreticalCoder:
    """Identifies a core category and themes using LSA extractive summarization."""

    def code(self, fc: FirstCycleResult,
             categories: list[Category]) -> tuple[CoreCategory, list[Theme]]:
        if not categories:
            return self._empty_result()

        # Core category = highest-frequency category
        core_cat = categories[0]

        # Collect excerpts from top codes (top 30 by frequency)
        top_code_labels = sorted(
            fc.code_frequencies.items(), key=lambda x: x[1], reverse=True
        )[:30]
        excerpts: list[str] = []
        for label, _ in top_code_labels:
            for code in fc.all_codes.get(label, []):
                if code.excerpt:
                    excerpts.append(code.excerpt)

        # Deduplicate excerpts
        seen_ex: set[str] = set()
        unique_excerpts = []
        for ex in excerpts:
            norm = ex[:80].lower()
            if norm not in seen_ex:
                seen_ex.add(norm)
                unique_excerpts.append(ex)

        # Generate themes via LSA summarisation
        themes = self._extract_themes(unique_excerpts, categories, core_cat.name)

        theoretical_statement = (
            f"{core_cat.name} emerges as the central phenomenon, "
            f"connecting {', '.join(c.name for c in categories[1:4])}."
        )

        return (
            CoreCategory(
                name=core_cat.name,
                description=core_cat.description,
                related_categories=[c.name for c in categories[1:6]],
                theoretical_statement=theoretical_statement,
            ),
            themes,
        )

    def _extract_themes(self, excerpts: list[str],
                        categories: list[Category],
                        core_name: str) -> list[Theme]:
        if not excerpts:
            return []

        # Try sumy LSA summariser
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lsa import LsaSummarizer
            from sumy.nlp.stemmers import Stemmer
            from sumy.utils import get_stop_words

            corpus = " ".join(excerpts[:60])
            parser = PlaintextParser.from_string(corpus, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")
            summary_sentences = [str(s) for s in summarizer(parser.document, 5)]
        except Exception:
            # Fallback: take first unique sentence from each top excerpt
            summary_sentences = []
            for ex in excerpts[:10]:
                sents = re.split(r'(?<=[.!?])\s+', ex)
                for s in sents:
                    if len(s) > 30 and s not in summary_sentences:
                        summary_sentences.append(s)
                        break
            summary_sentences = summary_sentences[:5]

        themes: list[Theme] = []
        # Identify which categories each theme sentence relates to
        for i, sentence in enumerate(summary_sentences):
            if not sentence.strip():
                continue
            lower_sent = sentence.lower()
            related_cats = [
                c.name for c in categories
                if any(word in lower_sent
                       for word in c.name.lower().split()[:2])
            ]
            if not related_cats:
                related_cats = [core_name]

            # Manifest vs latent
            level = "latent" if any(h in lower_sent for h in _HEDGING) else "manifest"

            # Evidence: pick an excerpt containing words from the sentence
            sent_words = set(re.findall(r"\b\w{4,}\b", lower_sent))
            evidence = []
            for ex in excerpts:
                ex_words = set(re.findall(r"\b\w{4,}\b", ex.lower()))
                if len(sent_words & ex_words) >= 2:
                    evidence.append(ex[:100])
                    if len(evidence) >= 2:
                        break

            themes.append(Theme(
                statement=sentence,
                categories=related_cats,
                evidence=evidence,
                level=level,
            ))

        return themes

    def _empty_result(self) -> tuple[CoreCategory, list[Theme]]:
        return (
            CoreCategory(
                name="UNDETERMINED",
                description="Insufficient data to determine core category.",
                related_categories=[],
                theoretical_statement="",
            ),
            [],
        )
