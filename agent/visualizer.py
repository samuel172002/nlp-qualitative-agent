from __future__ import annotations
import os
from collections import defaultdict
from pathlib import Path

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyvis.network import Network

from shared.models import FirstCycleResult, SecondCycleResult

# Node colour by type
NODE_COLORS = {
    "segment": "#b0c4de",
    "code": "#90ee90",
    "category": "#ffa07a",
    "theme": "#dda0dd",
    "core_category": "#ffd700",
}

NODE_SIZES = {
    "segment": 10,
    "code": 20,
    "category": 35,
    "theme": 45,
    "core_category": 60,
}

# Edge colour + width by relationship type
_EDGE_STYLES: dict[str, dict] = {
    "causes":          {"color": "#ff6b6b", "width": 3, "label": "causes"},
    "enables":         {"color": "#51cf66", "width": 3, "label": "enables"},
    "constrains":      {"color": "#ff922b", "width": 3, "label": "constrains"},
    "influences":      {"color": "#74c0fc", "width": 3, "label": "influences"},
    "requires":        {"color": "#da77f2", "width": 3, "label": "requires"},
    "reflects":        {"color": "#ffd43b", "width": 3, "label": "reflects"},
    "leads_to":        {"color": "#adb5bd", "width": 2, "label": "leads to"},
    "has_code":        {"color": "#3a3a5a", "width": 1, "label": ""},
    "belongs_to":      {"color": "#3a3a5a", "width": 1, "label": "belongs to"},
    "supports_theme":  {"color": "#cc5de8", "width": 2, "label": "supports"},
    "relates_to_core": {"color": "#ffd700", "width": 2, "label": "relates to"},
}


class Visualizer:
    def export(self, graph: nx.DiGraph, output_dir: str,
               fc: FirstCycleResult | None = None,
               sc: SecondCycleResult | None = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self._export_pyvis(graph, output_dir)
        self._export_png(graph, output_dir)
        if fc and sc:
            self._export_cooccurrence(fc, sc, output_dir)

    # ------------------------------------------------------------------ #
    # Interactive HTML via PyVis — fully self-contained (no local assets)
    # ------------------------------------------------------------------ #
    def _export_pyvis(self, graph: nx.DiGraph, output_dir: str) -> None:
        # cdn_resources="in_line" embeds all JS/CSS into the HTML so the file
        # works on any device without needing the local lib/ folder.
        try:
            net = Network(height="750px", width="100%", directed=True,
                          bgcolor="#1a1a2e", font_color="#ffffff",
                          cdn_resources="in_line")
        except TypeError:
            # Older pyvis (<0.3.1) doesn't support cdn_resources
            net = Network(height="750px", width="100%", directed=True,
                          bgcolor="#1a1a2e", font_color="#ffffff")

        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "springLength": 200,
              "damping": 0.15
            },
            "stabilization": { "iterations": 150 }
          },
          "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
            "smooth": { "type": "dynamic" }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true
          }
        }
        """)

        for node, attrs in graph.nodes(data=True):
            ntype = attrs.get("node_type", "code")
            color = NODE_COLORS.get(ntype, "#cccccc")
            size = NODE_SIZES.get(ntype, 20)
            freq = attrs.get("frequency", 1)
            size = size + min(freq * 2, 30)
            label = str(node)[:30]
            title = _build_tooltip(node, attrs)
            net.add_node(str(node), label=label, color=color, size=size,
                         title=title, font={"size": 12})

        for src, dst, attrs in graph.edges(data=True):
            etype = attrs.get("edge_type", "")
            style = _EDGE_STYLES.get(etype, {"color": "#555577", "width": 1, "label": etype})
            display_label = style["label"]
            net.add_edge(
                str(src), str(dst),
                title=f"<b>{display_label or etype}</b><br>{attrs.get('description', '')}",
                label=display_label,
                width=style["width"],
                font={"size": 9, "color": style["color"]},
                color={"color": style["color"], "highlight": "#ffffff"},
            )

        out_path = os.path.join(output_dir, "knowledge_graph.html")
        try:
            net.write_html(out_path, notebook=False, open_browser=False)
        except TypeError:
            net.save_graph(out_path)

    # ------------------------------------------------------------------ #
    # Static PNG — categories + themes subgraph only
    # ------------------------------------------------------------------ #
    def _export_png(self, graph: nx.DiGraph, output_dir: str) -> None:
        important_types = {"category", "theme", "core_category"}
        nodes = [n for n, d in graph.nodes(data=True)
                 if d.get("node_type") in important_types]
        if not nodes:
            nodes = list(graph.nodes())[:50]

        subgraph = graph.subgraph(nodes)
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")

        try:
            pos = nx.kamada_kawai_layout(subgraph)
        except Exception:
            pos = nx.spring_layout(subgraph, seed=42)

        node_colors = [NODE_COLORS.get(subgraph.nodes[n].get("node_type", "code"), "#cccccc")
                       for n in subgraph.nodes()]
        node_sizes = [NODE_SIZES.get(subgraph.nodes[n].get("node_type", "code"), 300) * 15
                      for n in subgraph.nodes()]

        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_edges(subgraph, pos, ax=ax, edge_color="#555577",
                               arrows=True, arrowsize=15, alpha=0.7)
        nx.draw_networkx_labels(subgraph, pos, ax=ax,
                                font_color="white", font_size=8,
                                labels={n: str(n)[:20] for n in subgraph.nodes()})

        patches = [mpatches.Patch(color=c, label=t) for t, c in NODE_COLORS.items()
                   if t != "segment"]
        ax.legend(handles=patches, loc="upper left",
                  facecolor="#2a2a4a", labelcolor="white", fontsize=9)
        ax.set_title("Qualitative Code Map", color="white", fontsize=14, pad=12)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "code_map.png"), dpi=150,
                    bbox_inches="tight", facecolor="#1a1a2e")
        plt.close()

    # ------------------------------------------------------------------ #
    # Category co-occurrence heatmap
    # ------------------------------------------------------------------ #
    def _export_cooccurrence(self, fc: FirstCycleResult,
                              sc: SecondCycleResult, output_dir: str) -> None:
        if not sc.categories:
            return

        cat_names = [c.name for c in sc.categories]
        code_to_cat = {}
        for cat in sc.categories:
            for code_label in cat.codes:
                code_to_cat[code_label] = cat.name

        matrix = defaultdict(lambda: defaultdict(int))
        for cs in fc.coded_segments:
            cats_in_seg = set()
            for code in cs.codes:
                cat = code_to_cat.get(code.label)
                if cat:
                    cats_in_seg.add(cat)
            for a in cats_in_seg:
                for b in cats_in_seg:
                    matrix[a][b] += 1

        import numpy as np
        n = len(cat_names)
        mat = np.zeros((n, n))
        for i, a in enumerate(cat_names):
            for j, b in enumerate(cat_names):
                mat[i, j] = matrix[a][b]

        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([c[:20] for c in cat_names], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([c[:20] for c in cat_names], fontsize=8)
        plt.colorbar(im, ax=ax, label="Co-occurrence count")
        ax.set_title("Category Co-occurrence Heatmap", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cooccurrence_heatmap.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()


def _build_tooltip(node: str, attrs: dict) -> str:
    lines = [f"<b>{node}</b>"]
    for k, v in attrs.items():
        if k != "text":
            lines.append(f"{k}: {v}")
    if "text" in attrs:
        lines.append(f"text: {attrs['text'][:100]}…")
    return "<br>".join(lines)
