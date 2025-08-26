"""Minimal stub of PocketFlow classes for local testing.

This lightweight implementation provides just enough behaviour to execute the
example flows in this repository.  It supports the small subset of features
used in the tests: sequential node execution with optional conditional
transitions and a simple batch flow helper.

It is **not** a drop-in replacement for the real PocketFlow package but allows
running the demo pipeline without the external dependency.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class _Transition:
    """Helper to support ``node - 'action' >> next`` syntax."""

    def __init__(self, node: "Node", action: str) -> None:
        self.node = node
        self.action = action

    def __rshift__(self, other: "Node") -> "Node":
        self.node._transitions[self.action] = other
        return other


class Node:
    """Basic executable unit.

    Subclasses are expected to override ``prep``, ``exec`` and ``post``.  The
    ``run`` method orchestrates these steps and routes to the next node based on
    the value returned from ``post``.
    """

    def __init__(self, max_retries: int = 1, wait: int = 0, **params: Any) -> None:
        self.max_retries = max_retries
        self.wait = wait
        self.params = params
        self._transitions: Dict[str, "Node"] = {}

    # wiring ------------------------------------------------------------
    def __rshift__(self, other: "Node") -> "Node":
        self._transitions["default"] = other
        return other

    def __sub__(self, action: str) -> _Transition:
        return _Transition(self, action)

    # execution --------------------------------------------------------
    def run(self, shared: Dict[str, Any]) -> str:
        prep_res = self.prep(shared)
        exec_res = self.exec(prep_res)
        action = self.post(shared, prep_res, exec_res)
        next_node = self._transitions.get(action) or self._transitions.get("default")
        if next_node:
            return next_node.run(shared)
        return action

    # default hooks ----------------------------------------------------
    def prep(self, shared: Dict[str, Any]) -> Any:  # pragma: no cover - interface
        return None

    def exec(self, prep_res: Any) -> Any:  # pragma: no cover - interface
        return None

    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Any) -> str:  # pragma: no cover - interface
        return "default"


class Flow(Node):
    """Container that orchestrates a sub-flow of nodes."""

    def __init__(self, start: Node) -> None:
        super().__init__()
        self.start = start

    def run(self, shared: Dict[str, Any]) -> str:
        prep_res = self.prep(shared)
        self.start.run(shared)
        action = self.post(shared, prep_res, None)
        next_node = self._transitions.get(action) or self._transitions.get("default")
        if next_node:
            return next_node.run(shared)
        return action


class BatchFlow(Flow):
    """Repeatedly run the start node with different parameters."""

    def prep(self, shared: Dict[str, Any]) -> List[Dict[str, Any]]:  # pragma: no cover - interface
        return []

    def run(self, shared: Dict[str, Any]) -> str:  # pragma: no cover - simple logic
        items = self.prep(shared)
        for params in items:
            self.start.params = params
            self.start.run(shared)
        action = self.post(shared, items, None)
        next_node = self._transitions.get(action) or self._transitions.get("default")
        if next_node:
            return next_node.run(shared)
        return action


__all__ = ["Node", "Flow", "BatchFlow"]
