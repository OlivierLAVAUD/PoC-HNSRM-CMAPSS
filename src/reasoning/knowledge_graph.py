"""Knowledge graph for neuro-symbolic explainability on FD001.

Builds a directed graph linking sensors -> rules -> state -> anomaly, so a
human reviewer can trace *which sensor evidence fired which rule and led to
the current decision*. Optional ``networkx`` / ``matplotlib`` dependency:
the module imports cleanly without them and ``visualize`` is a no-op with a
warning when they are missing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import logging

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import networkx as nx

    _GRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    nx = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    _GRAPH_AVAILABLE = False


# FD001 sensor groupings used to label sensor nodes in human-readable terms.
SENSOR_GROUPS: Dict[str, List[str]] = {
    "temperature": ["s3", "s4", "s9", "s14"],
    "pressure": ["s7", "s11", "s12"],
    "vibration": ["s2", "s13"],
    "flow": ["s21"],
    "misc": ["s8", "s15", "s17", "s20"],
}


class KnowledgeGraphGenerator:
    """Build and render a small neuro-symbolic knowledge graph.

    Nodes
    -----
    ``sensor:<name>``   one per FD001 sensor, annotated with its latest value.
    ``rule:<name>``     one per active / learned rule.
    ``state``           the current binary decision (0 = normal, 1 = failure).
    ``anomaly``         anomaly severity and confidence.

    Edges
    -----
    sensor -> rule    (sensor evidence implicated in the rule)
    rule   -> state   (rule contributes to the decision)
    state  -> anomaly (decision materialises as an anomaly signal)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.graph: Any = None
        if not _GRAPH_AVAILABLE:
            self.logger.warning(
                "networkx/matplotlib not available; KnowledgeGraphGenerator "
                "will record state but cannot render."
            )

    def _new_graph(self) -> Any:
        if not _GRAPH_AVAILABLE:
            return None
        return nx.DiGraph()

    def update_graph(
        self,
        current_state: Dict[str, Any],
        insights: Iterable[str],
        rules: Iterable[Dict[str, Any]],
        anomalies: Dict[str, Any],
    ) -> Any:
        """Build (or rebuild) the knowledge graph for one decision."""
        if not _GRAPH_AVAILABLE:
            self.logger.debug("Knowledge graph skipped (networkx unavailable).")
            return None

        g = self._new_graph()
        state_value = int(current_state.get("system_state", 0))

        # ---- anomaly node -------------------------------------------------
        severity = int(anomalies.get("severity", state_value))
        confidence = float(anomalies.get("confidence", 0.0))
        g.add_node(
            "anomaly",
            label=f"anomaly\nsev={severity}\nconf={confidence:.2f}",
            kind="anomaly",
            severity=severity,
            confidence=confidence,
        )

        # ---- state node ---------------------------------------------------
        g.add_node(
            "state",
            label=f"state\n{'failure' if state_value else 'normal'}",
            kind="state",
            value=state_value,
        )
        g.add_edge("state", "anomaly", kind="materialises_as")

        # ---- sensor nodes -------------------------------------------------
        sensor_values: Dict[str, float] = {}
        for key, value in current_state.items():
            if key in ("system_state",):
                continue
            try:
                sensor_values[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

        for name, value in sensor_values.items():
            group = self._sensor_group(name)
            g.add_node(
                f"sensor:{name}",
                label=f"{name}\n{value:.2f}\n({group})",
                kind="sensor",
                group=group,
                value=value,
            )
            # every sensor points to the state (evidence -> decision)
            g.add_edge(f"sensor:{name}", "state", kind="evidence_for")

        # ---- rule nodes ---------------------------------------------------
        for rule in rules or []:
            name = str(rule.get("rule", rule))
            confidence = float(rule.get("confidence", 0.0))
            g.add_node(
                f"rule:{name}",
                label=f"rule\n{name}\nconf={confidence:.2f}",
                kind="rule",
                confidence=confidence,
            )
            g.add_edge(f"rule:{name}", "state", kind="contributes_to")
            # link active rules to the sensors named in the rule text
            for sensor_name in sensor_values:
                if sensor_name in name:
                    g.add_edge(f"sensor:{sensor_name}", f"rule:{name}",
                               kind="implicates")

        # ---- insight annotations ------------------------------------------
        for idx, insight in enumerate(list(insights)[:8]):
            node_id = f"insight:{idx}"
            g.add_node(
                node_id,
                label=f"insight\n{insight}",
                kind="insight",
            )
            g.add_edge(node_id, "state", kind="explains")

        self.graph = g
        self.logger.info(
            "Knowledge graph updated: %d nodes, %d edges.",
            g.number_of_nodes(),
            g.number_of_edges(),
        )
        return g

    @staticmethod
    def _sensor_group(name: str) -> str:
        for group, members in SENSOR_GROUPS.items():
            if name in members:
                return group
        return "other"

    def visualize(
        self,
        output_path: str,
        plot_type: str = "focused_ieee",
    ) -> Optional[str]:
        """Render the knowledge graph to a PNG file.

        Parameters
        ----------
        output_path:
            Destination PNG path.
        plot_type:
            Kept for API compatibility with the original ANSR-DT KG
            (``full`` / ``focused_ieee`` / ``publication``). All variants
            render the same graph with a clean, publication-friendly style.

        Returns
        -------
        The output path, or ``None`` when graph rendering is unavailable.
        """
        if not _GRAPH_AVAILABLE or self.graph is None:
            self.logger.warning(
                "Knowledge graph visualization skipped "
                "(networkx/matplotlib unavailable or graph not built)."
            )
            return None

        import os

        g = self.graph
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(g, seed=42, k=0.9)

        kind_colors = {
            "sensor": "#4c78a8",
            "rule": "#f58518",
            "state": "#e45756",
            "anomaly": "#72b7b2",
            "insight": "#54a24b",
        }
        node_colors = [
            kind_colors.get(g.nodes[n].get("kind", "sensor"), "#999999")
            for n in g.nodes
        ]
        node_sizes = [
            900 if g.nodes[n].get("kind") in ("state", "anomaly") else 600
            for n in g.nodes
        ]

        nx.draw_networkx_nodes(
            g, pos, node_color=node_colors, node_size=node_sizes,
            linewidths=1.0, edgecolors="black",
        )
        nx.draw_networkx_edges(
            g, pos, arrows=True, arrowsize=14, edge_color="#8c8c8c",
            width=1.4, connectionstyle="arc3,rad=0.1",
        )
        labels = {n: g.nodes[n].get("label", n) for n in g.nodes}
        nx.draw_networkx_labels(
            g, pos, labels, font_size=8, font_family="sans-serif",
        )

        plt.title(
            "Neuro-symbolic knowledge graph (sensors -> rules -> state -> anomaly)",
            fontsize=11,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.logger.info("Knowledge graph visualization saved to %s", output_path)
        return output_path

