"""
tool_retriever.py

RAG-based retrieval of the most relevant MCP tools for a given task.

The full MCP tool set has 120 tools covering 9 application namespaces.
Including all 120 in every policy prompt would waste context and confuse the
model.  This module:
  1. Loads the tool registry (tools/tools_registry.json).
  2. For each new task, embeds the task description and all tool descriptions.
  3. Returns the top-K most similar tools (default K=15-20).

Embedding backend is configurable:
  - "bm25"  – fast keyword-based, no GPU required (default for low-resource)
  - "dense" – sentence-transformers cosine similarity (better quality)
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent.parent / "tools" / "tools_registry.json"
DEFAULT_TOP_K = 18


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

class ToolRetriever:
    """
    Retrieve the most relevant MCP tools for a task description.

    Parameters
    ----------
    registry_path : str | Path
        Path to tools_registry.json.
    top_k : int
        Maximum number of tools to return.
    backend : "bm25" | "dense"
        Retrieval algorithm.
    model_name : str
        Only used when backend="dense"; sentence-transformers model name.
    """

    def __init__(
        self,
        registry_path: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        backend: str = "bm25",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.top_k = top_k
        self.backend = backend
        self._tools: List[Dict[str, Any]] = []
        self._encoder = None

        path = Path(registry_path) if registry_path else REGISTRY_PATH
        self._load_registry(path)

        if backend == "dense":
            self._init_dense_encoder(model_name)

    # ------------------------------------------------------------------

    def retrieve(
        self,
        task_description: str,
        app_hint: Optional[str] = None,
        exclude_apps: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return up to top_k tool schemas most relevant to *task_description*.

        Parameters
        ----------
        task_description : str
            Natural-language task the agent must complete.
        app_hint : str | None
            If provided, boost tools belonging to this application (e.g. "libreoffice_calc").
        exclude_apps : list | None
            Tool application categories to exclude entirely.
        """
        if not self._tools:
            logger.warning("Tool registry is empty; returning empty list")
            return []

        # Pre-filter by excluded apps
        candidates = self._tools
        if exclude_apps:
            excl = {e.lower() for e in exclude_apps}
            candidates = [t for t in candidates if t.get("app", "").lower() not in excl]

        # Hard app filter: when active app is known, only expose tools from that
        # app (and its "_2" extension class, e.g. libreoffice_calc + libreoffice_calc2).
        # Falls back to all candidates if the hint matches nothing.
        if app_hint:
            hint_lower = app_hint.lower()
            app_matchers = {hint_lower, hint_lower + "2"}
            filtered = [t for t in candidates if t.get("app", "").lower() in app_matchers]
            if filtered:
                candidates = filtered

        if self.backend == "dense":
            scored = self._score_dense(task_description, candidates)
        else:
            scored = self._score_bm25(task_description, candidates)

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [t for _, t in scored[: self.top_k]]

        # Force-include high-value generic tools that every task should be able
        # to call (save / env_info / get_workbook_info). OSWorld-MCP uses the
        # same MANDATORY_KEYWORDS list. Without them models can never "close"
        # a task properly.
        _MANDATORY_KEYWORDS = ("save", "env_info", "get_workbook_info")
        selected_names = {t.get("name", "") for t in selected}
        for t in candidates:
            name = t.get("name", "")
            if name in selected_names:
                continue
            if any(kw in name for kw in _MANDATORY_KEYWORDS):
                selected.append(t)
                selected_names.add(name)

        logger.debug(
            "ToolRetriever: returned %d/%d tools for task=%r",
            len(selected), len(candidates), task_description[:60],
        )
        return selected

    def format_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        Render a tool list as the compact text block injected into the policy prompt.
        """
        lines: List[str] = ["Available MCP tools (use ACTION_TYPE: mcp to call):"]
        for i, t in enumerate(tools, 1):
            name = t.get("name", "")
            app = t.get("app", "")
            desc = t.get("description", "No description")
            params = t.get("params", {})
            param_str = ", ".join(
                f"{k}: {v.get('type', 'any')}" if isinstance(v, dict) else f"{k}: any"
                for k, v in params.items()
            )
            lines.append(f"  {i:>3}. [{app}] {name}({param_str})")
            lines.append(f"       {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Registry loading
    # ------------------------------------------------------------------

    def _load_registry(self, path: Path) -> None:
        if not path.exists():
            logger.warning("tools_registry.json not found at %s; tool retrieval disabled", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Format: either a flat list, or {app_name: [tool, ...]}
        if isinstance(data, list):
            self._tools = data
        elif isinstance(data, dict):
            for app, tools in data.items():
                for t in tools:
                    self._tools.append({**t, "app": app})
        else:
            logger.error("Unexpected tools_registry.json format")

        logger.info("Loaded %d tools from registry", len(self._tools))

    # ------------------------------------------------------------------
    # BM25 scorer
    # ------------------------------------------------------------------

    def _score_bm25(
        self, query: str, tools: List[Dict[str, Any]]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Simple BM25-ish TF-IDF scoring (no external library required)."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return [(0.0, t) for t in tools]

        # Build DF across all tool docs
        df: Dict[str, int] = defaultdict(int)
        docs: List[List[str]] = []
        for t in tools:
            doc = _tool_text(t)
            tokens = _tokenize(doc)
            docs.append(tokens)
            for tok in set(tokens):
                df[tok] += 1

        N = len(tools)
        k1, b, avg_dl = 1.5, 0.75, max(1, sum(len(d) for d in docs) / N)

        scored = []
        for t, doc_tokens in zip(tools, docs):
            dl = len(doc_tokens)
            tf_map: Dict[str, int] = defaultdict(int)
            for tok in doc_tokens:
                tf_map[tok] += 1
            score = 0.0
            for tok in query_tokens:
                if tok not in df:
                    continue
                idf = math.log((N - df[tok] + 0.5) / (df[tok] + 0.5) + 1)
                tf = tf_map.get(tok, 0)
                tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
                score += idf * tf_norm
            scored.append((score, t))
        return scored

    # ------------------------------------------------------------------
    # Dense scorer
    # ------------------------------------------------------------------

    def _init_dense_encoder(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(model_name)
            # Pre-embed all tools
            texts = [_tool_text(t) for t in self._tools]
            self._tool_embeddings = self._encoder.encode(texts, convert_to_numpy=True)
            logger.info("Dense encoder initialised (%s)", model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; falling back to BM25"
            )
            self.backend = "bm25"

    def _score_dense(
        self, query: str, tools: List[Dict[str, Any]]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        if self._encoder is None:
            return self._score_bm25(query, tools)
        import numpy as np
        q_emb = self._encoder.encode([query], convert_to_numpy=True)[0]
        scored = []
        for i, t in enumerate(tools):
            # tools may be a subset after exclusion filtering; look up by tool identity
            idx = self._tools.index(t) if t in self._tools else -1
            if idx >= 0:
                emb = self._tool_embeddings[idx]
                cos = float(
                    np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-9)
                )
            else:
                cos = 0.0
            scored.append((cos, t))
        return scored


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _tool_text(tool: Dict[str, Any]) -> str:
    parts = [
        tool.get("name", ""),
        tool.get("app", ""),
        tool.get("description", ""),
    ]
    for v in tool.get("params", {}).values():
        if isinstance(v, dict):
            parts.append(v.get("description", ""))
    return " ".join(p for p in parts if p)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
