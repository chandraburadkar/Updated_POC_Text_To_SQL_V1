# app/agents/query_rewriter.py
from __future__ import annotations

from app.audit.langsmith_tracing import traceable_fn

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.agents.llm_factory import get_llm

logger = logging.getLogger(__name__)


# -----------------------------
# Structured Output Schema
# -----------------------------
class QueryRewriteOutput(BaseModel):
    rewritten_query: str = Field(
        ..., description="Clear, SQL-friendly version of the user question"
    )
    intent: str = Field(
        ..., description="User intent: KPI | TREND | RANKING | ROOT_CAUSE | ANOMALY | UNKNOWN"
    )
    entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted entities like airport, date range, metric, top_n, granularity"
    )
    clarification_needed: bool = Field(
        default=False, description="True if user question is ambiguous"
    )
    clarification_question: str = Field(
        default="", description="Follow-up question if clarification is needed"
    )
    notes: str = Field(
        default="", description="Internal reasoning notes (for audit/debug)"
    )


def _format_chat_history(chat_history: Optional[List[Dict[str, str]]], max_turns: int = 8) -> str:
    """
    chat_history expected format:
      [{"role": "user"/"assistant", "content": "..."}, ...]
    """
    if not chat_history:
        return "(none)"
    recent = chat_history[-max_turns:]
    lines: List[str] = []
    for t in recent:
        role = (t.get("role") or "user").strip().lower()
        content = (t.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines) if lines else "(none)"


# -----------------------------
# Prompt Template
# -----------------------------
def _rewrite_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an airport operations analytics expert.

Your task:
1) Understand the user's question (including follow-ups)
2) Rewrite it into a precise analytics question
3) Identify intent and entities
4) Decide if clarification is needed

Follow-up handling:
- If user says "same airport", "same metric", "last month", "compare it", etc.,
  resolve it using chat history + memory_entities.
- If critical info is missing AND cannot be inferred, ask ONE clarification question.

Intent labels allowed:
- KPI, TREND, RANKING, ROOT_CAUSE, ANOMALY, UNKNOWN

Output rules:
- Respond STRICTLY in JSON using the provided schema.
- Do NOT output SQL.
- Entities should be best-effort JSON.
""".strip(),
            ),
            (
                "human",
                "User question:\n{query}\n\n"
                "Recent chat history:\n{chat_history}\n\n"
                "Memory entities (JSON) from last successful turn:\n{memory_entities}\n\n"
                "{format_instructions}",
            ),
        ]
    )


# -----------------------------
# Public Function
# -----------------------------
@traceable_fn("query_rewriter")
def rewrite_query(
    user_query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    memory_entities: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Runs the Query Rewriter Agent.
    Backward compatible:
      rewrite_query("...") works as before.

    Enhanced:
      rewrite_query("same airport last month", chat_history=[...], memory_entities={...})
    """

    llm = get_llm(temperature=temperature)

    parser = PydanticOutputParser(pydantic_object=QueryRewriteOutput)
    prompt = _rewrite_prompt(parser)

    chat_text = _format_chat_history(chat_history)
    mem = memory_entities or {}

    try:
        chain = prompt | llm | parser

        result: QueryRewriteOutput = chain.invoke(
            {
                "query": user_query,
                "chat_history": chat_text,
                "memory_entities": json.dumps(mem, ensure_ascii=False),
                "format_instructions": parser.get_format_instructions(),
            }
        )

        output = result.model_dump()
        # hardening defaults
        output["rewritten_query"] = (output.get("rewritten_query") or user_query).strip()
        output["intent"] = (output.get("intent") or "UNKNOWN").strip() or "UNKNOWN"
        output["entities"] = output.get("entities") or {}
        output["clarification_needed"] = bool(output.get("clarification_needed", False))
        output["clarification_question"] = (output.get("clarification_question") or "").strip()
        output["notes"] = (output.get("notes") or "").strip()
        output["timestamp_utc"] = datetime.utcnow().isoformat()
        return output

    except Exception as e:
        logger.error("Query rewrite failed", exc_info=True)

        return {
            "rewritten_query": user_query,
            "intent": "UNKNOWN",
            "entities": mem or {},
            "clarification_needed": False,
            "clarification_question": "",
            "notes": f"Rewrite failed: {e}",
            "timestamp_utc": datetime.utcnow().isoformat(),
        }