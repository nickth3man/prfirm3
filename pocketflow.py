from __future__ import annotations
from typing import Any, Dict, Optional


class Node:
    """Minimal PocketFlow-compatible node for tests.

    Each node implements the prep -> exec -> post lifecycle and stores
    transitions to successor nodes keyed by action labels.
    """

    def __init__(self, max_retries: int | None = None, wait: int | None = None):
        self.max_retries = max_retries or 0
        self.wait = wait or 0
        # Mapping of action label -> successor node
        self._transitions: Dict[Optional[str], Node] = {}

    # Overridable lifecycle hooks -------------------------------------------------
    def prep(self, shared: Dict[str, Any]) -> Any:  # pragma: no cover - defaults
        return None

    def exec(self, prep_res: Any) -> Any:  # pragma: no cover - defaults
        return None

    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:  # pragma: no cover
        return None

    # Wiring helpers --------------------------------------------------------------
    def __rshift__(self, other: "Node") -> "Node":
        """Connect default transition to ``other``."""
        self._transitions[None] = other
        return other

    def __sub__(self, label: str) -> "_LabeledEdge":
        return _LabeledEdge(self, label)

    # Execution ------------------------------------------------------------------
    def run(self, shared: Dict[str, Any]) -> Optional[str]:
        prep_res = self.prep(shared)
        exec_res = self.exec(prep_res)
        return self.post(shared, prep_res, exec_res)


class _LabeledEdge:
    def __init__(self, node: Node, label: str):
        self.node = node
        self.label = label

    def __rshift__(self, other: Node) -> Node:
        self.node._transitions[self.label] = other
        return other


class Flow(Node):
    """Simplified Flow orchestrator that is also a Node."""

    def __init__(self, start: Node, **kwargs: Any):
        super().__init__(**kwargs)
        self.start = start

    def run(self, shared: Dict[str, Any]) -> None:  # type: ignore[override]
        # Run flow-level prep hook
        self.prep(shared)

        node: Optional[Node] = self.start
        while node is not None:
            action = node.run(shared)
            node = node._transitions.get(action) or node._transitions.get(None)

        # Flow-level post hook receives no exec result by design
        self.post(shared, None, None)


class BatchFlow(Flow):
    """Very small batch flow implementation used for tests."""

    def prep(self, shared: Dict[str, Any]) -> list[Dict[str, Any]]:  # pragma: no cover
        return []

    def run(self, shared: Dict[str, Any]) -> None:  # type: ignore[override]
        params_list = self.prep(shared)
        for params in params_list:
            # Expose current params on start node for compatibility with existing
            # node implementations that expect ``self.params``.
            setattr(self.start, "params", params)
            node: Optional[Node] = self.start
            while node is not None:
                action = node.run(shared)
                node = node._transitions.get(action) or node._transitions.get(None)
        self.post(shared, None, None)
