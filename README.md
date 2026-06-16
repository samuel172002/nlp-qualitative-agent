# NLP Qualitative Coding Agent

A fully offline, LLM-free qualitative coding pipeline built on Saldaña's two-cycle coding framework. Upload interview transcripts (`.txt`, `.pdf`, `.docx`), and the system automatically applies 13 qualitative coding methods, builds a knowledge graph, and exports a PDF research report — all using local NLP libraries, no API keys required.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
  - [First Cycle — 9 Coding Methods](#first-cycle--9-coding-methods)
  - [Second Cycle — 4 Coding Methods](#second-cycle--4-coding-methods)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Usage Guide](#usage-guide)
- [Output Files](#output-files)
- [Data Model](#data-model)
- [Architecture](#architecture)

---

## Overview

Qualitative research involves reading interview transcripts and manually tagging passages with codes, then grouping those codes into categories, and finally identifying themes and a core theoretical finding. This process is rigorous but time-consuming.

This project automates every stage of that workflow using deterministic local NLP — no GPT, no Gemini, no API calls. Every coding decision is explainable and reproducible.

The methodology follows **Johnny Saldaña's *The Coding Manual for Qualitative Researchers*** (3rd ed.), which is the standard reference in social science research.

---

## Features

- **Zero LLMs** — runs 100% offline using spaCy, KeyBERT, VADER, and scikit-learn
- **13 qualitative coding methods** across two cycles (9 first, 4 second)
- **Interactive knowledge graph** — explore how codes connect to categories and themes
- **PDF report export** — publication-ready research summary
- **Multi-format input** — `.txt`, `.pdf`, `.docx` all supported
- **Streamlit UI** — clean dark-purple interface with 6 analysis tabs
- **Configurable clustering** — adjust the number of thematic clusters (3–15)
- **Research question alignment** — structural coding maps segments to your stated RQs

---

## How It Works

The pipeline has 7 sequential stages:

```
File Loading → First Cycle Coding → Second Cycle Coding
    → Knowledge Graph → Visualizations → Summary → PDF Export
```

### First Cycle — 9 Coding Methods

First-cycle coding assigns raw codes to individual text segments. Each of the 9 methods targets a different dimension of meaning:

| Method | What it captures | NLP technique |
|---|---|---|
| **Descriptive** | What is literally present in the text | Noun chunks ranked by corpus TF-IDF |
| **In Vivo** | Participants' own exact words and phrases | KeyBERT keyphrases (BERT embeddings, 2–4 ngrams) |
| **Process** | Actions, activities, and ongoing experiences | VBG gerunds detected via spaCy POS tags + inflect |
| **Initial** | Open-ended first impressions of each sentence | Top noun + top verb per sentence (spaCy) |
| **Structural** | How segments relate to research questions | Sentence-BERT cosine similarity vs. RQ embeddings |
| **Emotion** | Emotional content and affective tone | NRCLex (8 emotion categories) + VADER compound score |
| **Values** | Beliefs, attitudes, and value expressions | Curated lexicon lookup (values/beliefs/attitude adjectives) |
| **Versus** | Contradictions, tensions, and binary oppositions | Adversative conjunction detection + dependency tree parsing |
| **Evaluation** | Judgements, opinions, and assessments | VADER sentiment per sentence + syntactic subject extraction |

**Key design:** All heavy models (spaCy `en_core_web_sm`, BERT via sentence-transformers, KeyBERT) are loaded once via an `NLPResources` singleton and shared across all 9 coders, keeping memory usage low.

### Second Cycle — 4 Coding Methods

Second-cycle coding works on the output of first-cycle coding to find patterns and build theory:

| Method | What it does | NLP technique |
|---|---|---|
| **Pattern Coding** | Groups first-cycle codes into meta-patterns | TF-IDF vectorization + Agglomerative Clustering (cosine) |
| **Focused Coding** | Filters clusters to the most salient categories | Frequency thresholding (min. 2 occurrences) + TF-IDF term naming |
| **Axial Coding** | Finds causal/relational links between categories | Segment co-occurrence counting + verb-based relationship inference |
| **Theoretical Coding** | Identifies the core category and overarching themes | sumy LSA extractive summarization |

---

## Tech Stack

| Library | Role |
|---|---|
| `streamlit` | Web UI |
| `spacy` (`en_core_web_sm`) | POS tagging, dependency parsing, NER |
| `keybert` | BERT-based keyphrase extraction |
| `sentence-transformers` (`all-MiniLM-L6-v2`) | Semantic similarity for structural coding |
| `vaderSentiment` | Rule-based sentiment analysis |
| `nrclex` | NRC Emotion Lexicon (8 emotion types) |
| `inflect` | Verb → gerund conversion for process coding |
| `scikit-learn` | TF-IDF vectorization + agglomerative clustering |
| `sumy` | LSA extractive summarization for theoretical coding |
| `networkx` | Knowledge graph construction |
| `pyvis` | Interactive HTML knowledge graph visualization |
| `matplotlib` | Static code map and co-occurrence heatmap |
| `reportlab` | PDF report generation |
| `pypdf` | PDF file reading |
| `python-docx` | Word document reading |
| `numpy` | Numerical operations |

---

## Project Structure

```
nlp-qualitative-agent/
│
├── app.py                        # Streamlit UI — main entry point
├── nlp_pipeline.py               # Orchestrates all 7 pipeline stages
├── requirements.txt
│
├── shared/
│   └── models.py                 # All dataclasses (TextSegment, Code, Category, etc.)
│
├── agent/
│   ├── file_loader.py            # Reads .txt/.pdf/.docx, chunks into segments
│   ├── knowledge_graph.py        # Builds NetworkX directed graph
│   ├── visualizer.py             # PyVis HTML + matplotlib PNG exports
│   └── pdf_exporter.py           # ReportLab PDF report
│
├── nlp_first_cycle/
│   ├── coders.py                 # 9 coder classes + NLPResources singleton
│   └── engine.py                 # FirstCycleEngine — runs all 9 coders over segments
│
├── nlp_second_cycle/
│   ├── coders.py                 # PatternCoder, FocusedCoder, AxialCoder, TheoreticalCoder
│   └── engine.py                 # SecondCycleEngine — runs all 4 coders in sequence
│
├── lib/                          # PyVis static assets (vis-network, tom-select)
│
└── .streamlit/
    └── config.toml               # Dark purple Streamlit theme
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/samuel172002/nlp-qualitative-agent.git
cd nlp-qualitative-agent
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the spaCy language model**

```bash
python -m spacy download en_core_web_sm
```

That's it. No API keys, no accounts, no internet connection needed at runtime.

---

## Running the App

> **Important:** On Windows, use `python -m streamlit` instead of the `streamlit` command directly to avoid PATH issues.

```bash
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## Usage Guide

**1. Upload transcripts**
- Drag and drop one or more `.txt`, `.pdf`, or `.docx` files into the file uploader.
- These should be interview transcripts or other qualitative text data.

**2. Set research questions (optional)**
- Enter your research questions in the sidebar text area, one per line.
- These are used by the Structural Coder to align segments with your study focus.

**3. Adjust cluster count (optional)**
- Use the sidebar slider to set the number of thematic clusters (3–15, default 8).
- Higher values = more granular categories. Lower values = broader themes.

**4. Run Analysis**
- Click the **Run Analysis** button.
- Progress updates appear as each pipeline stage completes.
- Depending on transcript length, this takes roughly 30–90 seconds.

**5. Explore results across 6 tabs**

| Tab | Contents |
|---|---|
| **First Cycle Codes** | Frequency table of all codes, bar chart of top 20 |
| **Categories** | Expandable cards for each category with member codes, properties, dimensions |
| **Relationships** | Axial relationship table (source → type → target) |
| **Themes & Theory** | LSA-extracted themes, core category, theoretical statement |
| **Visualizations** | Interactive knowledge graph (PyVis), code map, co-occurrence heatmap |
| **Download** | PDF report, full JSON export, interactive HTML graph |

---

## Output Files

All outputs are saved to the `output/` directory:

| File | Description |
|---|---|
| `knowledge_graph.html` | Interactive network graph (open in any browser) |
| `code_map.png` | Static matplotlib visualization of code structure |
| `cooccurrence_heatmap.png` | Heatmap of how often categories co-occur in the same segment |
| `report.pdf` | Full PDF research report with all codes, categories, and theoretical findings |

---

## Data Model

The pipeline uses a clean dataclass hierarchy defined in `shared/models.py`:

```
TextSegment           — a chunk of text from a source file
    └── CodedSegment  — a segment with all its assigned codes
            └── Code  — a single code (label, type, excerpt, confidence)

FirstCycleResult
    ├── coded_segments   list[CodedSegment]
    ├── code_frequencies dict[str, int]
    └── all_codes        dict[str, list[Code]]

SecondCycleResult
    ├── categories         list[Category]
    ├── axial_relationships list[AxialRelationship]
    ├── themes             list[Theme]
    ├── core_category      CoreCategory
    └── pattern_codes      dict[str, list[str]]
```

**Code types** (`CodeType` enum):
`DESCRIPTIVE` · `IN_VIVO` · `PROCESS` · `INITIAL` · `STRUCTURAL` · `EMOTION` · `VALUES` · `VERSUS` · `EVALUATION` · `PATTERN`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          app.py (Streamlit UI)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    NLPPipeline.run()
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   FileLoader          FirstCycle          SecondCycle
   .load()             Engine.run()        Engine.run()
          │                  │                  │
  list[TextSegment]   ┌──────┴──────┐    ┌──────┴───────────┐
                      │ 9 Coders:   │    │ 4 Coders:        │
                      │ Descriptive │    │ PatternCoder     │
                      │ InVivo      │    │ FocusedCoder     │
                      │ Process     │    │ AxialCoder       │
                      │ Initial     │    │ TheoreticalCoder │
                      │ Structural  │    └──────────────────┘
                      │ Emotion     │
                      │ Values      │    KnowledgeGraphBuilder
                      │ Versus      │    Visualizer
                      │ Evaluation  │    PDFExporter
                      └─────────────┘
```

All NLP models are loaded once at startup via `NLPResources` (singleton pattern) and shared across all coders to avoid redundant loading of large model files.
