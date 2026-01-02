# app/graph/text2sql_graph.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import hashlib
import json

import pandas as pd

from app.services.ttl_cache import TTLCache
from app.audit.langsmith_tracing import tracing_session, traceable_fn
from app.state.agent_state import AgentState

from app.agents.query_rewriter import rewrite_query
from app.agents.sql_generator import generate_sql
from app.agents.sql_validator import validate_and_autofix_sql
from app.agents.sql_executor import execute_sql
from app.agents.explainer import explain_answer


# -----------------------------
# CACHES
# -----------------------------
REWRITE_CACHE = TTLCache(ttl_seconds=900, max_items=500)
PLAN_CACHE = TTLCache(ttl_seconds=600, max_items=500)


def _sha(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


@traceable_fn("run_text2sql")
def run_text2sql(
    user_question: str,
    top_k_schema: int = 5,
    return_rows: int = 20,
    enable_viz: bool = False,
    chat_history: Optional[List[Dict[str, str]]] = None,
    memory_entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    with tracing_session():
        state = AgentState(user_question=user_question)
        state.chat_history = chat_history or []
        state.memory_entities = memory_entities or {}

        # -----------------------------
        # STEP 5: Rewrite (cached)
        # -----------------------------
        rewrite_key = _sha({"q": _normalize_text(user_question)})
        rew = REWRITE_CACHE.get(rewrite_key)

        if not rew:
            rew = rewrite_query(
                user_query=user_question,
                chat_history=state.chat_history,
                memory_entities=state.memory_entities,
                temperature=0.0,
            )
            REWRITE_CACHE.set(rewrite_key, rew)

        if rew.get("clarification_needed"):
            return {
                "ok": False,
                "stage": "clarification",
                "message": rew.get("clarification_question"),
            }

        state.rewritten_query = rew.get("rewritten_query", user_question)
        state.intent = rew.get("intent", "UNKNOWN")
        state.entities = rew.get("entities", {})

        # -----------------------------
        # STEP 4: Schema (MCP-first)
        # -----------------------------
        import os
        use_mcp_schema = os.getenv("USE_MCP_SCHEMA", "0").lower() in ("1", "true", "yes")

        if use_mcp_schema:
            from app.mcp.bridge_client import mcp_get_schema
            from app.rag.mcp_schema_picker import pick_top_k_tables

            schema_json = mcp_get_schema("main")
            state.schema_context, state.retrieved_tables = pick_top_k_tables(
                schema_json=schema_json,
                query=state.rewritten_query,
                k=top_k_schema,
            )
        else:
            from app.rag.schema_index import get_schema_index

            vs = get_schema_index()
            docs = vs.similarity_search(state.rewritten_query, k=top_k_schema)
            state.schema_context = "\n\n".join(d.page_content for d in docs)
            state.retrieved_tables = [
                d.metadata.get("table") for d in docs if d.metadata
            ]

        if not state.schema_context:
            return {"ok": False, "stage": "schema", "message": "Schema not found"}

        # -----------------------------
        # STEP 6: SQL generation
        # -----------------------------
        cand = generate_sql(
            rewritten_query=state.rewritten_query,
            schema_context=state.schema_context,
            intent=state.intent,
            entities=state.entities,
            user_question=user_question,
        )

        candidate_sql = cand.get("sql") if isinstance(cand, dict) else cand

        # -----------------------------
        # STEP 7: Validation
        # -----------------------------
        val = validate_and_autofix_sql(
            rewritten_query=state.rewritten_query,
            schema_context=state.schema_context,
            candidate_sql=candidate_sql,
            max_retries=1,
        )

        state.final_sql = val.get("final_sql", candidate_sql)

        # -----------------------------
        # STEP 8: Execution (MCP-aware)
        # -----------------------------
        exec_out = execute_sql(state.final_sql, limit_preview=return_rows)
        state.result_df = exec_out.get("df")

        # -----------------------------
        # STEP 9: Explanation
        # -----------------------------
        state.explanation = explain_answer(
            user_question=user_question,
            sql=state.final_sql,
            df=state.result_df,
        )

        return {
            "ok": True,
            "final_sql": state.final_sql,
            "preview_markdown": exec_out.get("preview_markdown"),
            "rows": exec_out.get("df", pd.DataFrame()).to_dict("records"),
            "row_count": exec_out.get("row_count"),
            "explanation": state.explanation,
            "executor": exec_out.get("executor"),
        }