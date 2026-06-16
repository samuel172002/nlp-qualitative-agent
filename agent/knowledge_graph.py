from __future__ import annotations
import networkx as nx

from shared.models import FirstCycleResult, SecondCycleResult


class KnowledgeGraphBuilder:
    """Builds a directed NetworkX graph from first- and second-cycle results."""

    def build(self, fc: FirstCycleResult, sc: SecondCycleResult) -> nx.DiGraph:
        G = nx.DiGraph()

        # --- code nodes ---
        for label, codes in fc.all_codes.items():
            freq = fc.code_frequencies.get(label, 0)
            code_type = codes[0].code_type.value if codes else "unknown"
            G.add_node(label, node_type="code", frequency=freq,
                       code_type=code_type, description=codes[0].description if codes else "")

        # --- segment → code edges ---
        for cs in fc.coded_segments:
            seg_id = cs.segment.segment_id
            G.add_node(seg_id, node_type="segment", text=cs.segment.text[:120])
            for code in cs.codes:
                G.add_edge(seg_id, code.label, edge_type="has_code",
                           confidence=code.confidence)

        # --- category nodes ---
        for cat in sc.categories:
            G.add_node(cat.name, node_type="category", frequency=cat.frequency,
                       description=cat.description)
            for code_label in cat.codes:
                if G.has_node(code_label):
                    G.add_edge(code_label, cat.name, edge_type="belongs_to")

        # --- axial relationship edges ---
        for rel in sc.axial_relationships:
            if rel.source_category in G and rel.target_category in G:
                G.add_edge(rel.source_category, rel.target_category,
                           edge_type=rel.relationship_type,
                           description=rel.description)

        # --- theme nodes ---
        for i, theme in enumerate(sc.themes):
            theme_id = f"THEME_{i+1}"
            G.add_node(theme_id, node_type="theme", statement=theme.statement,
                       level=theme.level)
            for cat_name in theme.categories:
                if cat_name in G:
                    G.add_edge(cat_name, theme_id, edge_type="supports_theme")

        # --- core category node ---
        if sc.core_category:
            cc = sc.core_category
            G.add_node(cc.name, node_type="core_category",
                       description=cc.description,
                       theoretical_statement=cc.theoretical_statement)
            for rel_cat in cc.related_categories:
                if rel_cat in G:
                    G.add_edge(rel_cat, cc.name, edge_type="relates_to_core")

        return G
