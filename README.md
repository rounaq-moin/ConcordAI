# ConcordAI

ConcordAI is a stateful AI conflict mediation system. It takes two opposing user perspectives, classifies the conflict, retrieves similar cases, generates balanced responses, checks safety and quality, and stores the mediation turn history.

The repository folder is named `people-pleasing-ai`, but the product name is **ConcordAI**.

## Stack

- **Frontend:** React + Vite
- **Backend:** FastAPI
- **Orchestration:** LangGraph
- **LLM:** Groq
- **Vector store:** ChromaDB
- **Database:** MySQL
- **Local models:** MiniLM embeddings, DeBERTa/zero-shot validation, Detoxify safety checks

## Main Features

- Two-person mediation flow with User A and User B inputs.
- Multi-turn conversation lifecycle using `conversation_id` and `turn`.
- Resolved-state locking with dual ratings and optional comments.
- RAG retrieval from seed scenarios and approved memory cases.
- Mode-based generation:
  - `quality`: reasoning + reconciliation + critic path.
  - `fast`: faster non-production path.
  - `fast_production`: production agent path with deterministic checks and fallback.
- Groq production failover with primary and backup keys.
- Safety checking, output validation, retry, fallback, and memory quality gates.
- Evaluation runners, trace runner, and poster-ready evaluation charts.

## Project Structure

```text
agents/                    Multi-agent logic
api/                       FastAPI app, auth, DB helpers
coordinator/               LangGraph pipeline
data/scenarios.json        Seed scenarios for retrieval/evaluation
evaluation/                Older evaluation helper package
frontend/                  React + Vite app
models/                    Pydantic/shared schemas
utils/                     LLM, scoring, sanitization helpers

backend_eval.py            Backend test runner
backend_trace.py           Transparent trace/debug runner
evaluation_metrics.py      ML-style metrics from saved eval JSON
honest_evaluation_audit.py Leakage-aware evaluation audit report
export_dashboard_png.py    High-resolution PNG dashboard exporter
test_cases.json            Main test suite
```

## Backend Setup

From the project root:

```powershell
cd 'C:\RES AI\people-pleasing-ai'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `.env`.

```env
GROQ_API_KEY=gsk_primary_key_here
GROQ_BACKUP_API_KEYS=gsk_backup_1_here,gsk_backup_2_here
GROQ_MODEL=llama-3.1-8b-instant
GROQ_FALLBACK_MODELS=llama-3.3-70b-versatile
GROQ_TIMEOUT=12
GROQ_ACCOUNT_COOLDOWN_SECONDS=1200
GROQ_PRODUCTION_MAX_ATTEMPTS=6

MAX_RETRIES=3
RAG_MIN_SCORE=0.60
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=concordai
```

Notes:

- Do not commit `.env`.
- `GROQ_BACKUP_API_KEYS` is optional.
- The app creates the `concordai` MySQL database and required tables on startup.
- ChromaDB data is stored locally under `data/conflict_store/`.

## Run Backend

Normal backend run:

```powershell
cd 'C:\RES AI\people-pleasing-ai'
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000/health
```

## Live Backend Logs

To run the backend and save live logs:

```powershell
cd 'C:\RES AI\people-pleasing-ai'
New-Item -ItemType Directory -Force .\logs | Out-Null
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload --log-level debug 2>&1 | Tee-Object -FilePath .\logs\backend-live.log
```

In another PowerShell window, watch the logs:

```powershell
cd 'C:\RES AI\people-pleasing-ai'
Get-Content .\logs\backend-live.log -Wait -Tail 100
```

Useful log lines:

```text
[api] Ready
[retriever] Already seeded (...)
[graph:<trace_id>] retrieve ...
[graph:<trace_id>] production type=...
[graph:<trace_id>] safety approved=...
[graph:<trace_id>] done status=...
groq_summary attempts=... success=... fallback_used=...
[groq_failover] all_keys_exhausted ...
```

## Warmup

Warm local models before a demo:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/warmup `
  -ContentType 'application/json' `
  -Body '{"preload_local_models":true,"ping_groq":false}'
```

## Frontend Setup

```powershell
cd 'C:\RES AI\people-pleasing-ai\frontend'
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

Build production frontend:

```powershell
cd 'C:\RES AI\people-pleasing-ai\frontend'
npm run build
```

Preview production build:

```powershell
cd 'C:\RES AI\people-pleasing-ai\frontend'
npm run preview
```

Or serve the built `dist` folder:

```powershell
cd 'C:\RES AI\people-pleasing-ai\frontend'
npm run serve:dist
```

## API Endpoints

### Health

```http
GET /health
```

Returns backend status, retriever status, model status, and process metrics.

### Auth

```http
POST /auth/signup
POST /auth/login
GET  /auth/me
POST /auth/logout
```

Authenticated routes use:

```text
Authorization: Bearer <token>
```

### Mediate

```http
POST /mediate
```

Example:

```json
{
  "text_a": "You took credit for my idea in the meeting.",
  "text_b": "I built on your idea and did most of the final work.",
  "conversation_id": "optional-session-id",
  "turn": 1,
  "mode": "fast_production"
}
```

Modes:

```text
quality
fast
fast_production
```

Possible state errors:

```json
{ "error": "STALE_TURN", "latest_turn": 3, "status": "active" }
```

```json
{ "error": "ALREADY_RESOLVED", "status": "resolved", "resolved_turn": 3 }
```

### Resolve Conversation

```http
POST /conversations/{conversation_id}/resolve
```

Example:

```json
{
  "user_a_rating": 5,
  "user_b_rating": 4,
  "user_a_comment": "Felt heard.",
  "user_b_comment": "The response was fair.",
  "note": "Resolved after first turn."
}
```

The backend snapshots the latest saved mediation turn from the database. It does not trust frontend response data.

### History

```http
GET /history
```

Returns lightweight saved mediation history with status and resolution metadata.

## Trace Runner

Use this when you need to inspect one case deeply without the frontend:

```powershell
cd 'C:\RES AI\people-pleasing-ai'
python backend_trace.py --mode fast_production --case-id T02 --show-raw
```

Save a trace:

```powershell
python backend_trace.py --mode fast_production --case-id T02 --show-raw --save traces\T02_trace.json
```

Useful modes:

```text
quality
fast
fast_production
```

## Evaluation

Run focused cases:

```powershell
python backend_eval.py --mode fast_production --ids T01 T02 T03 T04 T06 T09 T14 T20
```

Run all cases and save:

```powershell
python backend_eval.py --mode fast_production --save all_100_after.json
```

Run in batches to avoid Groq free-tier limits:

```powershell
python backend_eval.py --mode fast_production --ids T01 T02 T03 T04 T05 T06 T07 T08 T09 T10 --save b01.json
python backend_eval.py --mode fast_production --ids T11 T12 T13 T14 T15 T16 T17 T18 T19 T20 --save b02.json
python backend_eval.py --mode fast_production --ids T21 T22 T23 T24 T25 T26 T27 T28 T29 T30 --save b03.json
python backend_eval.py --mode fast_production --ids T31 T32 T33 T34 T35 T36 T37 T38 T39 T40 --save b04.json
python backend_eval.py --mode fast_production --ids T41 T42 T43 T44 T45 T46 T47 T48 T49 T50 --save b05.json
python backend_eval.py --mode fast_production --ids T51 T52 T53 T54 T55 T56 T57 T58 T59 T60 --save b06.json
python backend_eval.py --mode fast_production --ids T61 T62 T63 T64 T65 T66 T67 T68 T69 T70 --save b07.json
python backend_eval.py --mode fast_production --ids T71 T72 T73 T74 T75 T76 T77 T78 T79 T80 --save b08.json
python backend_eval.py --mode fast_production --ids T81 T82 T83 T84 T85 T86 T87 T88 T89 T90 --save b09.json
python backend_eval.py --mode fast_production --ids T91 T92 T93 T94 T95 T96 T97 T98 T99 T100 --save b10.json
```

Give Groq a few minutes between batches if you are on the free tier.

## Metrics And Reports

Generate ML-style metrics from saved eval results:

```powershell
python evaluation_metrics.py --results b01.json b02.json b03.json b04.json b05.json b06.json b07.json b08.json b09.json b10.json --out metrics_report_batches
```

Generate leakage-aware honest audit report:

```powershell
python honest_evaluation_audit.py --results b01.json b02.json b03.json b04.json b05.json b06.json b07.json b08.json b09.json b10.json --out honest_eval_report_batches
```

Generate poster-ready PNG charts:

```powershell
python export_dashboard_png.py
```

Important generated outputs:

```text
honest_eval_report_batches/summary.json
honest_eval_report_batches/task_metrics.csv
honest_eval_report_batches/per_class_metrics.csv
honest_eval_report_batches/mismatches.csv
honest_eval_report_batches/charts.html
honest_eval_report_batches/charts/concordai_dashboard_poster_3600x2520.png
honest_eval_report_batches/charts/poster_sections/
```

These are generated artifacts. Do not commit them unless you intentionally want to publish report snapshots.

## Groq Failover

Production calls use an ordered failover chain:

```text
primary key -> backup key 1 -> backup key 2 -> deterministic fallback
```

Behavior:

- Maximum three keys total.
- Keys are tried in order, not randomly load-balanced.
- Rate-limited keys enter per-key cooldown.
- `retry-after` is used when available.
- Timeouts do not disable keys.
- Auth failures disable only that key for the current process.
- If all Groq attempts fail, deterministic fallback still returns a safe mediation response.

Expected logs:

```text
groq_failover account=primary fingerprint=gsk_...ABCD model=... reason=timeout
groq_summary attempts=2 success=primary:llama-3.3-70b-versatile fallback_used=false
[groq_failover] all_keys_exhausted keys_configured=3 keys_skipped=2 keys_attempted=1 last_model=... falling_back_to=deterministic
```

Raw keys are never printed.

## Database Notes

MySQL stores:

```text
users
auth_tokens
mediation_history
conversation_statuses
conversation_resolutions
```

ChromaDB stores:

```text
seed scenarios
retrieved cases
approved memory cases
```

Resetting local vector memory:

```powershell
# Stop the backend first, then remove data/conflict_store manually if needed.
```

Do not delete database or vector-store folders during a demo unless you intentionally want a fresh state.

## Recommended Demo Flow

1. Start MySQL.
2. Start backend with live logging.
3. Open a second terminal with `Get-Content .\logs\backend-live.log -Wait -Tail 100`.
4. Start frontend.
5. Sign up or log in.
6. Submit User A and User B perspectives.
7. Continue one more turn or click resolve.
8. Save ratings/comments.
9. Show history and resolved badge.
10. Use `backend_trace.py` if you need to explain the pipeline.

## Troubleshooting

### Backend cannot connect to MySQL

Check:

```text
MYSQL_HOST
MYSQL_PORT
MYSQL_USER
MYSQL_PASSWORD
MYSQL_DATABASE
```

Then restart the backend.

### Frontend cannot reach backend

Check backend is running:

```text
http://127.0.0.1:8000/health
```

Check `CORS_ORIGINS` includes:

```text
http://localhost:5173
http://127.0.0.1:5173
```

### `backend-live.log` does not exist

Create it by starting the backend with `Tee-Object`:

```powershell
New-Item -ItemType Directory -Force .\logs | Out-Null
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload --log-level debug 2>&1 | Tee-Object -FilePath .\logs\backend-live.log
```

### Groq rate limit or timeout

Use smaller eval batches, wait between batches, or add backup keys in `.env`.

### Generated charts look blurry in a poster

Use the high-resolution PNG exporter:

```powershell
python export_dashboard_png.py
```

Then use:

```text
honest_eval_report_batches/charts/concordai_dashboard_poster_3600x2520.png
honest_eval_report_batches/charts/poster_sections/
```

## Quick Command Cheat Sheet

```powershell
# Backend
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Backend with logs
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload --log-level debug 2>&1 | Tee-Object -FilePath .\logs\backend-live.log

# Watch logs
Get-Content .\logs\backend-live.log -Wait -Tail 100

# Frontend
cd frontend
npm run dev

# Trace one case
python backend_trace.py --mode fast_production --case-id T02 --show-raw

# Eval focused cases
python backend_eval.py --mode fast_production --ids T01 T02 T03 T04 T06 T09 T14 T20

# Eval all cases
python backend_eval.py --mode fast_production --save all_100_after.json

# Honest audit
python honest_evaluation_audit.py --results b01.json b02.json b03.json b04.json b05.json b06.json b07.json b08.json b09.json b10.json --out honest_eval_report_batches

# Poster PNGs
python export_dashboard_png.py
```
