from __future__ import annotations

from typing import Any, Dict, Optional, List
from datetime import date, datetime
from decimal import Decimal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.graph.text2sql_graph import run_text2sql

router = APIRouter()


# -----------------------------
# Helpers: JSON-safe conversion
# -----------------------------
def _json_safe(v: Any) -> Any:
    if v is None:
        return None

    if isinstance(v, (datetime, date)):
        return v.isoformat()

    if isinstance(v, Decimal):
        return float(v)

    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore")

    try:
        import numpy as np  # type: ignore
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass

    return v


def _df_to_rows(df: Any, limit: int) -> List[Dict[str, Any]]:
    if df is None:
        return []
    try:
        rows = df.head(limit).to_dict(orient="records")
        return [{k: _json_safe(val) for k, val in r.items()} for r in rows]
    except Exception:
        return []


def _rows_json_safe(rows: Any) -> List[Dict[str, Any]]:
    """Make already-materialized rows JSON safe too."""
    if not isinstance(rows, list):
        return []
    safe: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            safe.append({k: _json_safe(v) for k, v in r.items()})
    return safe


def _normalize_chat_history(chat_history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    if not chat_history:
        return []
    out: List[Dict[str, str]] = []
    for t in chat_history[-40:]:
        role = (t.get("role") or "user").strip().lower()
        content = (t.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


# -----------------------------
# Request/Response models
# -----------------------------
class Text2SQLRequest(BaseModel):
    question: str = Field(..., description="User natural language question")
    top_k_schema: int = Field(5, ge=1, le=20)
    return_rows: int = Field(20, ge=1, le=500)
    enable_viz: bool = Field(False, description="Reserved for future chart generation")

    chat_history: Optional[List[Dict[str, str]]] = Field(default=None)
    memory_entities: Optional[Dict[str, Any]] = Field(default=None)


class Text2SQLResponse(BaseModel):
    ok: bool

    final_sql: Optional[str] = None
    intent: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None
    retrieved_tables: Optional[List[str]] = None

    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    rows: Optional[List[Dict[str, Any]]] = None

    preview_markdown: Optional[str] = None
    explanation: Optional[Dict[str, Any]] = None

    chat_history: Optional[List[Dict[str, str]]] = None
    memory_entities: Optional[Dict[str, Any]] = None

    cache_hit: Optional[bool] = None

    debug: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    stage: Optional[str] = None


@router.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@router.post("/text2sql", response_model=Text2SQLResponse)
def text2sql(req: Text2SQLRequest) -> Dict[str, Any]:
    try:
        chat_history = _normalize_chat_history(req.chat_history)
        memory_entities = req.memory_entities or {}

        out = run_text2sql(
            user_question=req.question,
            top_k_schema=req.top_k_schema,
            return_rows=req.return_rows,
            enable_viz=req.enable_viz,
            chat_history=chat_history,
            memory_entities=memory_entities,
        )

        cache_hit = bool(out.get("cache_hit", False))

        if out.get("ok"):
            df_pack = out.get("dataframe") or {}
            df = df_pack.get("df", None)

            cols: List[str] = list(df_pack.get("columns") or [])
            row_count = df_pack.get("row_count", None)

            rows: List[Dict[str, Any]] = []
            if df is not None:
                rows = _df_to_rows(df, limit=req.return_rows)
                try:
                    cols = list(df.columns)
                    row_count = int(df.shape[0])
                except Exception:
                    pass

            # fallback if graph ever returns rows directly
            if not rows and isinstance(out.get("rows"), list):
                rows = _rows_json_safe(out.get("rows"))

            preview_md = out.get("preview_markdown") or df_pack.get("preview_markdown")

            return {
                "ok": True,
                "final_sql": out.get("final_sql"),
                "intent": out.get("intent"),
                "entities": out.get("entities"),
                "retrieved_tables": out.get("retrieved_tables"),

                "row_count": row_count,
                "columns": cols,
                "rows": rows,
                "preview_markdown": preview_md,

                "explanation": out.get("explanation"),

                "chat_history": out.get("chat_history", chat_history),
                "memory_entities": out.get("memory_entities", memory_entities),

                "cache_hit": cache_hit,
                "debug": out.get("debug"),
            }

        return {
            "ok": False,
            "stage": out.get("stage"),
            "message": out.get("message"),
            "intent": out.get("intent"),
            "entities": out.get("entities"),
            "retrieved_tables": out.get("retrieved_tables"),

            "chat_history": out.get("chat_history", chat_history),
            "memory_entities": out.get("memory_entities", memory_entities),

            "cache_hit": cache_hit,
            "debug": out.get("debug"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))