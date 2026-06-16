"""
Streamlit UI for the NLP Qualitative Coding Agent.
Run with: streamlit run app.py
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Qualitative Coding Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Bold & Modern dark-purple theme ──────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #13111c;
    color: #ede9fe;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b2e 0%, #16132a 100%);
    border-right: 1px solid #3b3460;
  }
  [data-testid="stSidebar"] * { color: #ede9fe !important; }
  [data-testid="stSidebar"] hr { border-color: #3b3460; }

  /* ── Header / title ── */
  .app-header {
    background: linear-gradient(135deg, #1e1b2e 0%, #2d1b69 50%, #1e1b2e 100%);
    border: 1px solid #3b3460;
    border-radius: 12px;
    padding: 28px 32px 20px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .app-header::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, #8b5cf640 0%, transparent 70%);
    pointer-events: none;
  }
  .app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px;
  }
  .app-header p {
    color: #a78bfa;
    margin: 0;
    font-size: 0.95rem;
  }

  /* ── Progress bar ── */
  [data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #7c3aed, #ec4899) !important;
    border-radius: 4px;
  }
  [data-testid="stProgressBar"] > div {
    background-color: #2d1b69 !important;
    border-radius: 4px;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: #1e1b2e;
    border: 1px solid #3b3460;
    border-radius: 10px;
    padding: 16px;
    transition: border-color 0.2s;
  }
  [data-testid="metric-container"]:hover {
    border-color: #8b5cf6;
  }
  [data-testid="stMetricValue"] {
    color: #a78bfa !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
  }
  [data-testid="stMetricLabel"] {
    color: #c4b5fd !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* ── Tabs ── */
  [data-testid="stTabs"] [role="tablist"] {
    background: #1e1b2e;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #3b3460;
  }
  [data-testid="stTabs"] [role="tab"] {
    border-radius: 6px;
    color: #a78bfa;
    font-weight: 500;
    padding: 8px 16px;
    transition: all 0.2s;
  }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white !important;
  }
  [data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {
    background: #2d1b69;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.2s;
    box-shadow: 0 4px 15px #7c3aed44;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px #7c3aed66;
    background: linear-gradient(135deg, #6d28d9, #9333ea);
  }
  .stButton > button:disabled {
    background: #2d1b69;
    box-shadow: none;
    transform: none;
    opacity: 0.5;
  }

  /* ── Expanders ── */
  [data-testid="stExpander"] {
    background: #1e1b2e;
    border: 1px solid #3b3460;
    border-radius: 8px;
    margin-bottom: 8px;
  }
  [data-testid="stExpander"]:hover { border-color: #8b5cf6; }
  [data-testid="stExpanderToggleIcon"] { color: #8b5cf6 !important; }

  /* ── Dataframes ── */
  [data-testid="stDataFrame"] {
    border: 1px solid #3b3460;
    border-radius: 8px;
    overflow: hidden;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: #1e1b2e;
    border: 2px dashed #3b3460;
    border-radius: 10px;
    transition: border-color 0.2s;
  }
  [data-testid="stFileUploader"]:hover { border-color: #8b5cf6; }

  /* ── Info / success / error boxes ── */
  [data-testid="stAlert"] {
    border-radius: 8px;
    border-left-width: 4px;
  }

  /* ── Divider ── */
  hr { border-color: #3b3460 !important; }

  /* ── Subheaders ── */
  h2, h3 {
    background: linear-gradient(90deg, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* ── Sidebar title ── */
  .sidebar-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }

  /* ── Download buttons ── */
  .stDownloadButton > button {
    background: #1e1b2e;
    color: #a78bfa;
    border: 1px solid #3b3460;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s;
    width: 100%;
    margin-bottom: 8px;
  }
  .stDownloadButton > button:hover {
    background: #2d1b69;
    border-color: #8b5cf6;
    color: #ede9fe;
  }

  /* ── Slider ── */
  [data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">⚙ Settings</p>', unsafe_allow_html=True)
    st.markdown("---")

    n_clusters = st.slider(
        "Number of category clusters",
        min_value=3, max_value=15, value=8,
        help="Controls how many categories the second cycle generates. "
             "Increase for larger datasets.",
    )

    rq_input = st.text_area(
        "Research Questions (one per line)",
        placeholder="How do participants cope with barriers?\nWhat support systems do they use?",
        height=140,
        help="Optional. If provided, the Structural Coder will link segments to these questions.",
    )
    research_questions = [q.strip() for q in rq_input.splitlines() if q.strip()]

    st.markdown("---")
    st.caption(
        "Implements Saldaña's two-cycle coding framework using local NLP — "
        "no LLM or API key required."
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🔬 Qualitative Coding Agent</h1>
  <p>Upload interview transcripts for a full two-cycle qualitative analysis &mdash;
     no LLM or API key required. Accepts <strong>.txt</strong>, <strong>.pdf</strong>,
     and <strong>.docx</strong> files.</p>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload transcript files",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
)

run_btn = st.button("▶ Run Analysis", type="primary",
                    disabled=not uploaded_files)

# ── Session state ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "fc" not in st.session_state:
    st.session_state.fc = None
if "sc" not in st.session_state:
    st.session_state.sc = None
if "output_dir" not in st.session_state:
    st.session_state.output_dir = None


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn and uploaded_files:
    # Save uploaded files to a temp directory
    tmp_dir = tempfile.mkdtemp()
    file_paths: list[str] = []
    for uf in uploaded_files:
        dest = os.path.join(tmp_dir, uf.name)
        with open(dest, "wb") as f:
            f.write(uf.getbuffer())
        file_paths.append(dest)

    output_dir = os.path.join(tmp_dir, "output")

    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    TOTAL_STAGES = 7

    stage_messages: list[str] = []

    def on_progress(stage: int, total: int, message: str):
        stage_messages.append(message)
        pct = int((stage / total) * 100)
        progress_bar.progress(pct)
        status_placeholder.info(f"**Stage {stage}/{total}:** {message}")

    from nlp_pipeline import NLPPipeline
    pipeline = NLPPipeline()
    pipeline.add_progress_callback(on_progress)

    try:
        with st.spinner("Running analysis…"):
            from nlp_first_cycle.engine import FirstCycleEngine
            from nlp_second_cycle.engine import SecondCycleEngine
            from shared.models import FirstCycleResult, SecondCycleResult

            # We run manually so we can store fc/sc for tabs
            from agent.file_loader import FileLoader
            from agent.knowledge_graph import KnowledgeGraphBuilder
            from agent.visualizer import Visualizer
            from agent.pdf_exporter import PDFExporter
            import time

            os.makedirs(output_dir, exist_ok=True)
            start = time.time()

            on_progress(1, TOTAL_STAGES, "Loading and segmenting files…")
            segments = FileLoader().load(file_paths)

            on_progress(2, TOTAL_STAGES,
                        f"Running first-cycle coding on {len(segments)} segments…")
            fc_engine = FirstCycleEngine(research_questions=research_questions)
            fc_result: FirstCycleResult = fc_engine.run(segments)

            on_progress(3, TOTAL_STAGES, "Running second-cycle coding…")
            from nlp_first_cycle.coders import NLPResources
            nlp = NLPResources().nlp if NLPResources()._loaded else None
            sc_engine = SecondCycleEngine(n_clusters=n_clusters, spacy_nlp=nlp)
            sc_result: SecondCycleResult = sc_engine.run(fc_result)

            on_progress(4, TOTAL_STAGES, "Building knowledge graph…")
            graph = KnowledgeGraphBuilder().build(fc_result, sc_result)

            on_progress(5, TOTAL_STAGES, "Exporting visualizations…")
            Visualizer().export(graph, output_dir, fc=fc_result, sc=sc_result)

            on_progress(6, TOTAL_STAGES, "Building summary…")
            summary = pipeline._build_summary(
                file_paths, segments, fc_result, sc_result,
                elapsed=time.time() - start,
            )

            on_progress(7, TOTAL_STAGES, "Exporting PDF report…")
            pdf_path = PDFExporter().export(summary, fc_result, sc_result, output_dir)
            summary["pdf_path"] = pdf_path

        progress_bar.progress(100)
        status_placeholder.success(
            f"Analysis complete in {summary['elapsed_seconds']}s — "
            f"{summary['unique_code_labels']} unique codes across {summary['total_segments']} segments."
        )

        st.session_state.result = summary
        st.session_state.fc = fc_result
        st.session_state.sc = sc_result
        st.session_state.output_dir = output_dir

    except Exception as e:
        progress_bar.empty()
        status_placeholder.error(f"Pipeline error: {e}")
        st.exception(e)


# ── Results tabs ──────────────────────────────────────────────────────────────
if st.session_state.result:
    summary = st.session_state.result
    fc = st.session_state.fc
    sc = st.session_state.sc
    output_dir = st.session_state.output_dir

    st.markdown("---")

    # Metric cards
    cols = st.columns(5)
    metrics = [
        ("Segments", summary["total_segments"]),
        ("Unique Codes", summary["unique_code_labels"]),
        ("Categories", summary["category_count"]),
        ("Relationships", summary["relationship_count"]),
        ("Themes", summary["theme_count"]),
    ]
    for col, (label, val) in zip(cols, metrics):
        col.metric(label, val)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 First Cycle Codes",
        "🗂 Categories",
        "🔗 Relationships",
        "💡 Themes & Theory",
        "🕸 Visualizations",
        "📄 Download",
    ])

    # ── Tab 1: First Cycle Codes ─────────────────────────────────────────
    with tab1:
        st.subheader("Top First-Cycle Codes by Frequency")
        import pandas as pd
        rows = []
        for label, freq in sorted(fc.code_frequencies.items(),
                                  key=lambda x: x[1], reverse=True)[:80]:
            codes_list = fc.all_codes.get(label, [])
            ctype = codes_list[0].code_type.value if codes_list else ""
            excerpt = codes_list[0].excerpt[:80] if codes_list else ""
            rows.append({"Code": label, "Type": ctype,
                         "Frequency": freq, "Sample Excerpt": excerpt})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Frequency bar chart
        st.markdown("**Frequency Distribution (top 20)**")
        chart_data = df.head(20).set_index("Code")["Frequency"]
        st.bar_chart(chart_data)

    # ── Tab 2: Categories ────────────────────────────────────────────────
    with tab2:
        st.subheader("Second-Cycle Categories")
        for cat in sc.categories:
            with st.expander(f"**{cat.name}** — frequency {cat.frequency}"):
                st.write(f"**Description:** {cat.description}")
                st.write(f"**Member codes ({len(cat.codes)}):** "
                         f"{', '.join(cat.codes[:10])}"
                         f"{'…' if len(cat.codes) > 10 else ''}")
                if cat.properties:
                    st.write(f"**Properties:** {', '.join(cat.properties)}")
                if cat.dimensions:
                    st.write(f"**Dimensions:** {', '.join(cat.dimensions)}")

    # ── Tab 3: Axial Relationships ───────────────────────────────────────
    with tab3:
        st.subheader("Axial Relationships Between Categories")
        if sc.axial_relationships:
            rel_rows = [
                {
                    "Source": r.source_category,
                    "Relationship": r.relationship_type,
                    "Target": r.target_category,
                    "Description": r.description,
                }
                for r in sc.axial_relationships
            ]
            st.dataframe(pd.DataFrame(rel_rows),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No relationships found — try providing more data or "
                    "lowering the co-occurrence threshold.")

    # ── Tab 4: Themes & Theory ───────────────────────────────────────────
    with tab4:
        if sc.core_category:
            st.subheader("Core Category")
            st.markdown(f"### {sc.core_category.name}")
            st.write(sc.core_category.description)
            if sc.core_category.theoretical_statement:
                st.info(sc.core_category.theoretical_statement)
            if sc.core_category.related_categories:
                st.write(f"**Related:** {', '.join(sc.core_category.related_categories)}")

        if sc.themes:
            st.subheader("Themes")
            for i, theme in enumerate(sc.themes, 1):
                level_badge = "🔵 Manifest" if theme.level == "manifest" else "🟣 Latent"
                st.markdown(f"**Theme {i}** {level_badge}")
                st.write(theme.statement)
                if theme.evidence:
                    with st.expander("Evidence"):
                        for ev in theme.evidence:
                            st.caption(f"• {ev}")
                st.markdown("---")

    # ── Tab 5: Visualizations ────────────────────────────────────────────
    with tab5:
        html_path = os.path.join(output_dir, "knowledge_graph.html")
        png_path = os.path.join(output_dir, "code_map.png")
        heatmap_path = os.path.join(output_dir, "cooccurrence_heatmap.png")

        if os.path.exists(html_path):
            st.subheader("Interactive Knowledge Graph")
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=750, scrolling=False)

        col_a, col_b = st.columns(2)
        with col_a:
            if os.path.exists(png_path):
                st.subheader("Code Map")
                st.image(png_path, use_container_width=True)
        with col_b:
            if os.path.exists(heatmap_path):
                st.subheader("Co-occurrence Heatmap")
                st.image(heatmap_path, use_container_width=True)

    # ── Tab 6: Download ──────────────────────────────────────────────────
    with tab6:
        st.subheader("Download Results")
        pdf_path = summary.get("pdf_path", "")
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📄 Download PDF Report",
                    data=f,
                    file_name="analysis_report.pdf",
                    mime="application/pdf",
                )

        # JSON summary
        import json
        json_summary = {k: v for k, v in summary.items() if k != "pdf_path"}
        st.download_button(
            "📊 Download Summary JSON",
            data=json.dumps(json_summary, indent=2),
            file_name="analysis_summary.json",
            mime="application/json",
        )

        # HTML graph
        if os.path.exists(html_path):
            with open(html_path, "rb") as f:
                st.download_button(
                    "🕸 Download Knowledge Graph (HTML)",
                    data=f,
                    file_name="knowledge_graph.html",
                    mime="text/html",
                )
