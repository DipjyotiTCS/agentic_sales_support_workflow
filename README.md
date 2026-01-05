# Agentic Sales & Support Workflow (Flask + LangGraph + SQLite + KB)

This is a **demo-ready** agentic ecosystem for handling Sales and Support enquiries from email.
It loads your provided CSV datasets into **SQLite** at startup, provides a **chat-like UI** for sales reps,
and supports **Knowledge Base (PDF) ingestion** with chunking + embeddings + retrieval.

## Features
- Classification agent → routes to Sales or Support workflows
- Ticket logging (SQLite)
- KB-first context retrieval for both flows
- Sales intents: product inquiry, requirements → recommendations, pricing/bundling (policy checked), order status, missing-info email drafting
- Support intents: auth check, access/license check, refund eligibility with evidence + human-approval routing
- Guardrails: input validation, injection heuristics, output schema validation
- Explainability/traceability: step-by-step trace with confidence + evidence, stored in audit tables

## Quickstart (Windows / macOS / Linux)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# set your OpenAI key (optional for rule-based demo mode)
# Windows PowerShell:
$env:OPENAI_API_KEY="YOUR_KEY"
# macOS/Linux:
export OPENAI_API_KEY="YOUR_KEY"

python app.py
```

Open: http://127.0.0.1:5000

## Data loading
On startup, the app loads the CSVs from the `data/` folder into SQLite tables:
- customers, products, subscriptions, orders, pricing_policies, refund_policies, tickets

## Knowledge Base
Upload PDFs from the Home page or via API:
- UI: upload on homepage
- API: `POST /api/kb/upload` (multipart form field name: `file`)

## API
- `POST /api/chat` JSON:
```json
{ "sender_email":"customer@x.com", "subject":"...", "body":"..." }
```

## Notes
- If `OPENAI_API_KEY` is not set, the system runs in **deterministic demo mode** (rules + SQL evidence).
- With an OpenAI key, the system uses GPT for classification + reasoning and uses OpenAI embeddings for KB retrieval.


## Configuration (.env)
Create a `.env` file (or copy `.env.example` to `.env`) and set:
- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL` (default: gpt-4o-mini)
- `OPENAI_EMBED_MODEL` (default: text-embedding-3-small)

The app loads `.env` automatically using `python-dotenv`.
