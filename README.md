# Updated_POC_Text_To_SQL_V1


GARV Amadeus Text2SQL — Runbook + Repo Guide

This is a repo-specific runbook for your current setup:
	•	MCP server (Starlette + FastMCP + DuckDB) runs on :9000
	•	FastAPI Text2SQL API runs on :8000
	•	Streamlit UI runs on :8501

You typically need 2 or 3 terminals:
	1.	MCP server, 2) FastAPI, 3) Streamlit (optional but recommended)

⸻

1) Repository layout (what each folder/file does)

Paths below are relative to your repo root: garv-amadeus-text2sql/

A) app/ (main application)

app/main.py
Purpose: Entry-point that can run:
	•	CLI mode (python -m app.main "question")
	•	API mode (python -m app.main --api)

Key functions:
	•	build_api() → creates FastAPI app and mounts routes
	•	run_api() → runs uvicorn

⸻

B) app/api/ (FastAPI routes)

app/api/routes.py
Purpose: Defines HTTP endpoints like:
	•	GET /api/health
	•	POST /api/text2sql

What it returns:
	•	final_sql
	•	rows (from SQL execution)
	•	preview_markdown
	•	explanation
	•	chat_history, memory_entities

⸻

C) app/graph/ (Text2SQL pipeline orchestration)

app/graph/text2sql_graph.py
Purpose: End-to-end orchestration pipeline:
	1.	rewrite query
	2.	fetch schema context
	3.	generate SQL
	4.	validate + autofix SQL
	5.	execute SQL
	6.	explain answer
	7.	update memory/chat history
	8.	caching

Important:
	•	Calls executor: app/agents/sql_executor.py
	•	Uses caching: app/services/ttl_cache.py

⸻

D) app/agents/ (LLM agent modules)

These are modular “agents” called by the graph.

app/agents/query_rewriter.py
Purpose: converts user input into a clean rewritten query + intent + entities.

app/agents/sql_generator.py
Purpose: generates candidate SQL based on:
	•	rewritten query
	•	schema context

app/agents/sql_validator.py
Purpose: validates SQL and optionally autofixes.

app/agents/sql_executor.py
Purpose: executes SQL either:
	•	via MCP bridge if USE_MCP=1
	•	or direct DuckDB fallback

app/agents/explainer.py
Purpose: creates explanation summary using returned dataframe + SQL.

⸻

E) app/mcp/ (MCP server + bridge)

app/mcp/server.py
Purpose: MCP server that exposes:
	•	/health
	•	/bridge/run_sql (POST)
	•	/bridge/get_schema (POST)
	•	/mcp (FastMCP protocol endpoint)

Governance included:
	•	API key auth via header x-mcp-api-key
	•	limits (MCP_MAX_ROWS, MCP_DEFAULT_LIMIT)
	•	blocked keywords (DDL/DML)
	•	audit logging (mcp_audit.log)

⸻

F) app/db/ (DuckDB connection)

app/db/duckdb_client.py
Purpose: provides get_conn() to DuckDB database.

⸻

G) app/rag/ (schema retrieval)

app/rag/schema_index.py
Purpose: builds/loads schema vector index (if you use vector retrieval).

If you’re currently using MCP schema directly, this may be bypassed.

⸻

H) app/services/ (utilities)

app/services/ttl_cache.py
Purpose: in-memory TTL cache for:
	•	rewrite cache
	•	plan cache

⸻

I) Streamlit UI

streamlit_app.py (or your actual file name)
Purpose: UI that calls:
	•	GET http://127.0.0.1:8000/api/health
	•	POST http://127.0.0.1:8000/api/text2sql

⸻

2) Environment setup (two venvs)

You have two environments:
	•	.venv_mcp → MCP server deps
	•	.venv → FastAPI + Streamlit deps

2.1 Create venvs (one-time)

From repo root:

python3 -m venv .venv
python3 -m venv .venv_mcp


⸻

3) Install dependencies + generate requirements

Option A (recommended): keep two requirements files

requirements_api.txt (FastAPI + app + Streamlit)
Activate .venv:

source .venv/bin/activate
pip install -U pip
pip install -r requirements_api.txt

requirements_mcp.txt (MCP server)
Activate .venv_mcp:

source .venv_mcp/bin/activate
pip install -U pip
pip install -r requirements_mcp.txt

Option B: Create requirements from current machine

Generate API requirements

source .venv/bin/activate
pip freeze > requirements_api.txt

Generate MCP requirements

source .venv_mcp/bin/activate
pip freeze > requirements_mcp.txt

Then on another machine:

pip install -r requirements_api.txt
pip install -r requirements_mcp.txt


⸻

4) How to run (exact order, with file names)

Terminal 1 — MCP server

cd garv-amadeus-text2sql
source .venv_mcp/bin/activate

export MCP_HOST=127.0.0.1
export MCP_PORT=9000
export MCP_API_KEY=test_key

PYTHONPATH=. python -m app.mcp.server

✅ Test:

curl http://127.0.0.1:9000/health


⸻

Terminal 2 — FastAPI server

cd garv-amadeus-text2sql
source .venv/bin/activate

export USE_MCP=1
export MCP_BASE_URL=http://127.0.0.1:9000
export MCP_API_KEY=test_key

PYTHONPATH=. python -m app.main --api --host 127.0.0.1 --port 8000

✅ Test:

curl http://127.0.0.1:8000/api/health


⸻

Terminal 3 — Streamlit UI

cd garv-amadeus-text2sql
source .venv/bin/activate

streamlit run streamlit_app.py

Open:
	•	http://localhost:8501

⸻

5) Sanity test notebook (Jupyter)

Run these in a notebook:

Cell A — quick helpers

import os, json, requests

MCP_BASE = os.getenv("MCP_BASE_URL", "http://127.0.0.1:9000")
API_BASE = "http://127.0.0.1:8000"

API_KEY = os.getenv("MCP_API_KEY", "")

def post_json(url, payload, headers=None, timeout=60):
    h = headers or {}
    h.setdefault("Content-Type", "application/json")
    if API_KEY:
        h.setdefault("x-mcp-api-key", API_KEY)
    return requests.post(url, json=payload, headers=h, timeout=timeout)

def pretty(r):
    print("status:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)

print("MCP_BASE:", MCP_BASE)
print("API_BASE:", API_BASE)
print("MCP_API_KEY set?:", bool(API_KEY))

Cell B — health checks

r1 = requests.get(f"{MCP_BASE}/health")
print("MCP /health:", r1.status_code)
print(r1.text)

r2 = requests.get(f"{API_BASE}/api/health")
print("API /api/health:", r2.status_code)
print(r2.text)

Cell C — run a Text2SQL query

question = "Top 5 airports by avg security wait time last 7 days"
payload = {
    "question": question,
    "top_k_schema": 5,
    "return_rows": 20,
    "enable_viz": False,
    "chat_history": [],
    "memory_entities": {},
}

r = post_json(f"{API_BASE}/api/text2sql", payload, timeout=180)
pretty(r)


⸻

6) Where to change ports / base URLs
	•	MCP port: set MCP_PORT (default 9000)
	•	API port: --port 8000
	•	Streamlit: defaults to 8501

FastAPI calls MCP using:
	•	MCP_BASE_URL=http://127.0.0.1:9000

MCP auth:
	•	set same MCP_API_KEY in both terminals

⸻

7) Troubleshooting quick map

MCP /health returns 500

Usually means:
	•	config YAML missing (if you enabled yaml-based governance)
	•	DuckDB file path issues
	•	syntax error in server.py

Fix: check MCP terminal logs. Ensure required config files exist.

API works but Text2SQL fails

Usually means:
	•	MCP server not running
	•	missing USE_MCP=1 or wrong MCP_BASE_URL

Streamlit shows “API Offline”

Start FastAPI first and confirm /api/health returns 200.

⸻

8) What to commit into GitHub

Recommended to commit:
	•	requirements_api.txt
	•	requirements_mcp.txt
	•	README.md (this runbook content)

Do NOT commit:
	•	.venv/, .venv_mcp/
	•	.env (store .env.example instead)

⸻

9) Suggested .env.example

Create .env.example in repo root:

# FastAPI
USE_MCP=1
MCP_BASE_URL=http://127.0.0.1:9000
MCP_API_KEY=test_key

# MCP Server
MCP_HOST=127.0.0.1
MCP_PORT=9000
MCP_MAX_ROWS=500
MCP_DEFAULT_LIMIT=50
MCP_AUDIT_LOG_PATH=./mcp_audit.log


⸻

If you want next, I can add a “File-by-file deep explanation” section where I explain:
	•	inputs/outputs of each function
	•	how state moves across agents
	•	where MCP is invoked
	•	how caching works
