"""
First-cycle NLP coders — 9 methods from Saldaña's Coding Manual.
All heavy models are loaded once via NLPResources and shared across coders.
"""
from __future__ import annotations
import re
from collections import defaultdict
from typing import Optional

import spacy
from spacy.language import Language
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import inflect

from shared.models import Code, CodeType, TextSegment


# ---------------------------------------------------------------------------
# Singleton resource loader
# ---------------------------------------------------------------------------

class NLPResources:
    """Loads all heavy NLP models once; pass this into every coder."""

    _instance: Optional["NLPResources"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self, research_questions: list[str] | None = None) -> None:
        if self._loaded:
            return

        # spaCy
        try:
            self.nlp: Language = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )

        # VADER
        self.vader = SentimentIntensityAnalyzer()

        # KeyBERT (lazy import to avoid slow cold-start if not installed)
        try:
            from keybert import KeyBERT
            self.keybert = KeyBERT()
        except ImportError:
            self.keybert = None

        # Sentence-BERT for structural coding
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
            self.np = np
        except ImportError:
            self.sbert = None
            self.np = None

        # Research question embeddings
        self.research_questions = research_questions or []
        if self.sbert and self.research_questions:
            self.rq_embeddings = self.sbert.encode(self.research_questions)
        else:
            self.rq_embeddings = []

        # inflect engine
        self.inflect_engine = inflect.engine()

        self._loaded = True


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_STOP_LABELS = {
    "THIS", "THAT", "THESE", "THOSE", "IT", "THEY", "HE", "SHE", "WE",
    "YOU", "I", "ME", "THEM", "HIM", "HER", "US", "ONE", "WHICH", "WHO",
}


def _clean_label(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9 ''\-]", "", text).strip().upper()
    return label[:60]


def _short_excerpt(text: str, phrase: str, window: int = 80) -> str:
    idx = text.lower().find(phrase.lower())
    if idx == -1:
        return text[:window]
    start = max(0, idx - window // 2)
    end = min(len(text), idx + len(phrase) + window // 2)
    return text[start:end].strip()


def _tfidf_scores(texts: list[str]) -> dict[str, float]:
    """Return word → tfidf score averaged across documents."""
    if not texts:
        return {}
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=200)
        mat = vec.fit_transform(texts)
        scores = dict(zip(vec.get_feature_names_out(),
                          mat.toarray().mean(axis=0).tolist()))
    except ValueError:
        scores = {}
    return scores


def _to_gerund(word: str, engine: inflect.engine) -> str:
    """Convert a verb to its gerund form if needed."""
    w = word.lower()
    if w.endswith("ing"):
        return word.upper()
    # simple heuristic
    if w.endswith("e") and len(w) > 2:
        return (w[:-1] + "ing").upper()
    return (w + "ing").upper()


# ---------------------------------------------------------------------------
# 1. Descriptive Coder
# ---------------------------------------------------------------------------

class DescriptiveCoder:
    """Labels the topic of a segment as noun phrases (UPPER CASE)."""

    def __init__(self, resources: NLPResources,
                 corpus_texts: list[str] | None = None):
        self.res = resources
        self.corpus_scores = _tfidf_scores(corpus_texts or [])

    def code(self, segment: TextSegment) -> list[Code]:
        doc = self.res.nlp(segment.text)
        chunks = list(doc.noun_chunks)
        if not chunks:
            return []

        # Score each chunk by average corpus TF-IDF weight of its tokens
        def chunk_score(chunk):
            words = [t.lemma_.lower() for t in chunk if not t.is_stop]
            if not words:
                return 0.0
            return sum(self.corpus_scores.get(w, 0.01) for w in words) / len(words)

        ranked = sorted(chunks, key=chunk_score, reverse=True)
        codes: list[Code] = []
        seen: set[str] = set()
        for chunk in ranked[:5]:
            label = _clean_label(chunk.text)
            if len(label) < 3 or label in _STOP_LABELS or label in seen:
                continue
            seen.add(label)
            excerpt = _short_excerpt(segment.text, chunk.text)
            score = chunk_score(chunk)
            codes.append(Code(
                label=label,
                code_type=CodeType.DESCRIPTIVE,
                description=f"Descriptive topic: {chunk.text}",
                excerpt=excerpt,
                confidence=min(score * 5, 0.95) if score > 0 else 0.5,
            ))
            if len(codes) >= 3:
                break
        return codes


# ---------------------------------------------------------------------------
# 2. In Vivo Coder
# ---------------------------------------------------------------------------

class InVivoCoder:
    """Extracts participant's own memorable phrases using KeyBERT."""

    def __init__(self, resources: NLPResources):
        self.res = resources

    def code(self, segment: TextSegment) -> list[Code]:
        if self.res.keybert is None:
            return self._fallback(segment)

        try:
            keyphrases = self.res.keybert.extract_keywords(
                segment.text,
                keyphrase_ngram_range=(2, 4),
                stop_words="english",
                top_n=5,
                use_maxsum=True,
                nr_candidates=15,
            )
        except Exception:
            return self._fallback(segment)

        codes: list[Code] = []
        for phrase, score in keyphrases:
            # Must contain a noun
            doc = self.res.nlp(phrase)
            has_noun = any(t.pos_ in ("NOUN", "PROPN") for t in doc)
            if not has_noun:
                continue
            label = _clean_label(phrase)
            if len(label) < 4 or label in _STOP_LABELS:
                continue
            excerpt = _short_excerpt(segment.text, phrase)
            codes.append(Code(
                label=label,
                code_type=CodeType.IN_VIVO,
                description=f"Participant's own phrase: '{phrase}'",
                excerpt=excerpt,
                confidence=float(score),
            ))
        return codes[:3]

    def _fallback(self, segment: TextSegment) -> list[Code]:
        """If KeyBERT unavailable, use high-TF-IDF noun+verb bigrams."""
        doc = self.res.nlp(segment.text)
        bigrams = []
        tokens = [t for t in doc if not t.is_stop and not t.is_punct and t.is_alpha]
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            has_noun = any(t.pos_ in ("NOUN", "PROPN") for t in [a, b])
            if has_noun:
                bigrams.append(f"{a.text} {b.text}")
        scores = _tfidf_scores([segment.text])
        ranked = sorted(bigrams,
                        key=lambda bg: scores.get(bg.lower(), 0),
                        reverse=True)
        codes = []
        for bg in ranked[:3]:
            label = _clean_label(bg)
            if len(label) < 4:
                continue
            codes.append(Code(
                label=label,
                code_type=CodeType.IN_VIVO,
                description=f"Salient phrase: '{bg}'",
                excerpt=_short_excerpt(segment.text, bg),
                confidence=0.6,
            ))
        return codes


# ---------------------------------------------------------------------------
# 3. Process Coder
# ---------------------------------------------------------------------------

class ProcessCoder:
    """Finds actions as gerunds (ADAPTING, NEGOTIATING) using VBG POS tags."""

    def __init__(self, resources: NLPResources):
        self.res = resources

    def code(self, segment: TextSegment) -> list[Code]:
        doc = self.res.nlp(segment.text)
        seen: set[str] = set()
        codes: list[Code] = []

        for token in doc:
            if token.tag_ == "VBG" and token.is_alpha and not token.is_stop:
                gerund = _to_gerund(token.lemma_, self.res.inflect_engine)
                if gerund in seen or len(gerund) < 5:
                    continue
                seen.add(gerund)
                # Get sentence context
                sent = token.sent.text
                codes.append(Code(
                    label=gerund,
                    code_type=CodeType.PROCESS,
                    description=f"Process action: {token.text} (gerund form)",
                    excerpt=sent[:120],
                    confidence=0.75,
                ))

        return codes[:5]


# ---------------------------------------------------------------------------
# 4. Initial Coder
# ---------------------------------------------------------------------------

class InitialCoder:
    """Open, line-by-line coding — one code per sentence."""

    def __init__(self, resources: NLPResources):
        self.res = resources

    def code(self, segment: TextSegment) -> list[Code]:
        doc = self.res.nlp(segment.text)
        codes: list[Code] = []

        for sent in doc.sents:
            sent_doc = sent.as_doc()
            # Top noun chunk or first noun
            noun_chunks = list(sent_doc.noun_chunks)
            best_noun = noun_chunks[0].text if noun_chunks else None
            if best_noun is None:
                nouns = [t.text for t in sent_doc if t.pos_ in ("NOUN", "PROPN")]
                best_noun = nouns[0] if nouns else None

            # Top verb (not auxiliary)
            verbs = [t.text for t in sent_doc
                     if t.pos_ == "VERB" and t.dep_ not in ("aux", "auxpass")]
            best_verb = verbs[0] if verbs else None

            parts = [p for p in [best_verb, best_noun] if p]
            if not parts:
                continue
            label = _clean_label(" ".join(parts))
            if len(label) < 3:
                continue

            codes.append(Code(
                label=label,
                code_type=CodeType.INITIAL,
                description=f"Initial impression: {sent.text[:60]}",
                excerpt=sent.text[:120],
                confidence=0.55,
            ))

        return codes


# ---------------------------------------------------------------------------
# 5. Structural Coder
# ---------------------------------------------------------------------------

class StructuralCoder:
    """Links segments to research questions using Sentence-BERT cosine similarity."""

    def __init__(self, resources: NLPResources):
        self.res = resources

    def code(self, segment: TextSegment) -> list[Code]:
        if not self.res.sbert or not self.res.research_questions:
            return self._keyword_fallback(segment)

        import numpy as np
        seg_emb = self.res.sbert.encode([segment.text])[0]
        rq_embs = self.res.rq_embeddings

        sims = []
        for i, rq_emb in enumerate(rq_embs):
            cos_sim = float(np.dot(seg_emb, rq_emb) /
                            (np.linalg.norm(seg_emb) * np.linalg.norm(rq_emb) + 1e-9))
            sims.append((i, cos_sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        codes = []
        for rq_idx, score in sims[:2]:
            if score < 0.2:
                continue
            rq = self.res.research_questions[rq_idx]
            label = _clean_label(f"RQ{rq_idx + 1} {rq[:30]}")
            codes.append(Code(
                label=label,
                code_type=CodeType.STRUCTURAL,
                description=f"Relates to RQ{rq_idx + 1}: {rq[:80]}",
                excerpt=segment.text[:120],
                confidence=score,
            ))
        return codes

    def _keyword_fallback(self, segment: TextSegment) -> list[Code]:
        if not self.res.research_questions:
            return []
        doc = self.res.nlp(segment.text)
        seg_words = {t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop}
        codes = []
        for i, rq in enumerate(self.res.research_questions):
            rq_doc = self.res.nlp(rq)
            rq_words = {t.lemma_.lower() for t in rq_doc if t.is_alpha and not t.is_stop}
            if not rq_words:
                continue
            overlap = len(seg_words & rq_words) / len(rq_words)
            if overlap > 0.15:
                label = _clean_label(f"RQ{i + 1} {rq[:30]}")
                codes.append(Code(
                    label=label,
                    code_type=CodeType.STRUCTURAL,
                    description=f"Keyword overlap with RQ{i + 1}: {rq[:80]}",
                    excerpt=segment.text[:120],
                    confidence=overlap,
                ))
        return codes[:2]


# ---------------------------------------------------------------------------
# 6. Emotion Coder
# ---------------------------------------------------------------------------

_EMOTION_THRESHOLD = 0.05   # minimum NRCLex frequency to create a code

class EmotionCoder:
    """Detects emotions via NRC Emotion Lexicon + VADER sentiment."""

    def __init__(self, resources: NLPResources):
        self.res = resources

    def code(self, segment: TextSegment) -> list[Code]:
        codes: list[Code] = []

        # NRCLex emotions
        try:
            nrc = NRCLex(segment.text)
            freqs = nrc.affect_frequencies
            dominant_emotions = sorted(
                [(k, v) for k, v in freqs.items() if v >= _EMOTION_THRESHOLD],
                key=lambda x: x[1], reverse=True
            )
            for emotion, score in dominant_emotions[:4]:
                if emotion in ("positive", "negative"):
                    continue
                label = _clean_label(emotion)
                codes.append(Code(
                    label=label,
                    code_type=CodeType.EMOTION,
                    description=f"NRC emotion detected: {emotion} (score={score:.2f})",
                    excerpt=segment.text[:120],
                    confidence=min(score * 2, 0.95),
                ))
        except Exception:
            pass

        # VADER overall sentiment
        vs = self.res.vader.polarity_scores(segment.text)
        compound = vs["compound"]
        if compound >= 0.3:
            codes.append(Code(
                label="POSITIVE SENTIMENT",
                code_type=CodeType.EMOTION,
                description=f"VADER compound={compound:.2f}",
                excerpt=segment.text[:120],
                confidence=abs(compound),
            ))
        elif compound <= -0.3:
            codes.append(Code(
                label="NEGATIVE SENTIMENT",
                code_type=CodeType.EMOTION,
                description=f"VADER compound={compound:.2f}",
                excerpt=segment.text[:120],
                confidence=abs(compound),
            ))

        return codes[:5]


# ---------------------------------------------------------------------------
# 7. Values Coder
# ---------------------------------------------------------------------------

_VALUES_LEXICON = {
    "freedom", "liberty", "justice", "equality", "fairness", "dignity",
    "respect", "integrity", "honesty", "loyalty", "family", "community",
    "security", "safety", "health", "education", "opportunity", "hope",
    "love", "compassion", "responsibility", "accountability", "faith",
    "tradition", "progress", "diversity", "inclusion", "solidarity",
}

_BELIEFS_MARKERS = {
    "think", "believe", "know", "feel", "trust", "assume", "suppose",
    "understand", "realize", "reckon", "consider", "regard", "view",
}

_ATTITUDE_ADJ = {
    "good", "bad", "great", "terrible", "wonderful", "awful", "excellent",
    "horrible", "important", "worthless", "necessary", "useless", "vital",
    "dangerous", "helpful", "harmful", "positive", "negative", "beneficial",
    "detrimental", "significant", "meaningless", "valuable", "pointless",
}


class ValuesCoder:
    """Identifies Values (V:), Attitudes (A:), Beliefs (B:) via lexicon lookup."""

    def __init__(self, resources: NLPResources):
        self.res = resources

    def code(self, segment: TextSegment) -> list[Code]:
        doc = self.res.nlp(segment.text)
        codes: list[Code] = []
        seen: set[str] = set()

        for token in doc:
            lemma = token.lemma_.lower()
            word = token.text.lower()
            sent_text = token.sent.text[:120]

            # Values
            if lemma in _VALUES_LEXICON and f"V:{lemma}" not in seen:
                seen.add(f"V:{lemma}")
                codes.append(Code(
                    label=f"V: {lemma.upper()}",
                    code_type=CodeType.VALUES,
                    description=f"Value expressed: {lemma}",
                    excerpt=sent_text,
                    confidence=0.75,
                ))

            # Beliefs — verb + following clause
            if lemma in _BELIEFS_MARKERS and token.pos_ == "VERB":
                key = f"B:{lemma}"
                if key not in seen:
                    seen.add(key)
                    codes.append(Code(
                        label=f"B: {lemma.upper()}",
                        code_type=CodeType.VALUES,
                        description=f"Belief marker: '{token.text}'",
                        excerpt=sent_text,
                        confidence=0.65,
                    ))

            # Attitudes — evaluative adjectives
            if word in _ATTITUDE_ADJ and token.pos_ == "ADJ":
                # Find what the adjective modifies
                head = token.head.text if token.head != token else ""
                key = f"A:{word}:{head}"
                if key not in seen:
                    seen.add(key)
                    target = head.upper() if head else "GENERAL"
                    codes.append(Code(
                        label=f"A: {word.upper()} {target}"[:40],
                        code_type=CodeType.VALUES,
                        description=f"Attitude: {word} toward {head or 'subject'}",
                        excerpt=sent_text,
                        confidence=0.65,
                    ))

        return codes[:6]


# ---------------------------------------------------------------------------
# 8. Versus Coder
# ---------------------------------------------------------------------------

_ADVERSATIVE = {"but", "however", "whereas", "yet", "although", "though",
                "while", "vs", "versus", "against", "unlike", "despite",
                "nevertheless", "nonetheless"}


class VersusCoder:
    """Finds conflicts/tensions using adversative conjunctions + dependency parsing."""

    def __init__(self, resources: NLPResources):
        self.res = resources

    def code(self, segment: TextSegment) -> list[Code]:
        doc = self.res.nlp(segment.text)
        codes: list[Code] = []
        seen: set[str] = set()

        for token in doc:
            if token.text.lower() not in _ADVERSATIVE:
                continue
            # Get left and right clause heads
            left = self._clause_head(token, "left", doc)
            right = self._clause_head(token, "right", doc)
            if not left or not right:
                continue
            label = _clean_label(f"{left} VS {right}")
            if label in seen or len(label) < 5:
                continue
            seen.add(label)
            codes.append(Code(
                label=label,
                code_type=CodeType.VERSUS,
                description=f"Tension: {left} vs {right} (connector: '{token.text}')",
                excerpt=token.sent.text[:150],
                confidence=0.70,
            ))

        return codes[:4]

    def _clause_head(self, conjunction_token, side: str, doc) -> str:
        idx = conjunction_token.i
        if side == "left":
            candidates = [t for t in doc[:idx]
                          if t.pos_ in ("NOUN", "PROPN", "VERB") and not t.is_stop]
            return candidates[-1].lemma_.upper() if candidates else ""
        else:
            candidates = [t for t in doc[idx + 1:]
                          if t.pos_ in ("NOUN", "PROPN", "VERB") and not t.is_stop]
            return candidates[0].lemma_.upper() if candidates else ""


# ---------------------------------------------------------------------------
# 9. Evaluation Coder
# ---------------------------------------------------------------------------

class EvaluationCoder:
    """Assigns +/−/0 polarity codes with opinion target using VADER + dependency."""

    def __init__(self, resources: NLPResources):
        self.res = resources

    def code(self, segment: TextSegment) -> list[Code]:
        doc = self.res.nlp(segment.text)
        codes: list[Code] = []
        seen: set[str] = set()

        for sent in doc.sents:
            vs = self.res.vader.polarity_scores(sent.text)
            compound = vs["compound"]
            if abs(compound) < 0.2:
                continue

            polarity = "+" if compound > 0 else "-"
            target = self._find_opinion_target(sent.as_doc())
            label = _clean_label(f"{polarity}{target}" if target else f"{polarity}OVERALL")
            if label in seen:
                continue
            seen.add(label)

            codes.append(Code(
                label=label,
                code_type=CodeType.EVALUATION,
                description=f"Evaluation: {polarity} polarity (compound={compound:.2f}), target={target or 'general'}",
                excerpt=sent.text[:120],
                confidence=min(abs(compound), 0.95),
            ))

        return codes[:4]

    def _find_opinion_target(self, sent_doc) -> str:
        """Find the noun nearest to the strongest sentiment word."""
        # Simple heuristic: subject of the sentence or first noun
        subjects = [t for t in sent_doc if t.dep_ in ("nsubj", "nsubjpass")]
        if subjects:
            return subjects[0].lemma_.upper()
        nouns = [t for t in sent_doc if t.pos_ in ("NOUN", "PROPN") and not t.is_stop]
        if nouns:
            return nouns[0].lemma_.upper()
        return ""
