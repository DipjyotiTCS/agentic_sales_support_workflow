import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from csv_loader import ensure_tables_from_csv
from kb import ingest_pdf
from conversation import new_conversation, add_message, get_messages, respond
from graph import build_graph
from guardrails import sanitize_text, guard_email_input
from schemas import EmailIn, ChatOut, TraceStep
from audit import new_run_id
from db import DB_PATH
import sqlite3

load_dotenv()

app = Flask(__name__)

GRAPH = build_graph()

# Flask 3.x removed `before_first_request`. Load CSV->SQLite at startup so the
# demo is deterministic.
stats = ensure_tables_from_csv()
app.logger.info(f"Loaded CSV tables into SQLite: {stats} (db={DB_PATH})")

@app.get("/api/health")
def health():
    return jsonify({"status":"ok", "db_path": DB_PATH})

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/kb/upload")
def kb_upload():
    if "file" not in request.files:
        return jsonify({"error":"Missing file field 'file'"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error":"Only PDF files supported"}), 400

    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, f.filename)
    f.save(save_path)

    doc_id, chunks = ingest_pdf(save_path, f.filename)
    return jsonify({"doc_id": doc_id, "chunks": chunks})

@app.post("/api/chat")
def chat():
    data = request.get_json(force=True)
    email_in = EmailIn(**data)

    subject = sanitize_text(email_in.subject)
    body = sanitize_text(email_in.body)
    ok, flags = guard_email_input(subject, body)

    run_id = new_run_id()
    state = {
        "run_id": run_id,
        "email": {
            "sender_email": str(email_in.sender_email),
            "subject": subject,
            "body": body if len(body) <= 20000 else body[:20000],
        },
        "guardrails": {"ok": ok, "flags": flags},
    }

    result = GRAPH.invoke(state)

    # Build response trace from audit table for this run_id
    trace = []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT step_name, confidence, evidence_json, output_json FROM agent_runs WHERE run_id = ? ORDER BY rowid ASC", (run_id,)).fetchall()
        for r in rows:
            evidence = []
            try:
                import json
                evidence = json.loads(r["evidence_json"] or "[]")
            except Exception:
                evidence = []
            outj = {}
            try:
                import json
                outj = json.loads(r["output_json"] or "{}")
            except Exception:
                outj = {}
            trace.append({
                "step": r["step_name"],
                "confidence": float(r["confidence"] or 0.0),
                "evidence": evidence,
                "reasoning": outj.get("rationale") or outj.get("reasoning") or "See evidence.",
            })
    finally:
        try:
            conn.close()
        except Exception:
            pass

    print("Output:", result)
    # Build assistant conversational message for the UI chat thread
    assistant_message = result.get("summary","").strip()
    parts = []
    if assistant_message:
        parts.append(assistant_message)
    if result.get("drafted_email"):
        parts.append("Draft reply you can send:\n" + result.get("drafted_email"))
    # Add follow-up question if guardrails flagged or missing info suggested
    if result.get("missing_info_questions"):
        qs = result.get("missing_info_questions")
        if isinstance(qs, list) and qs:
            parts.append("To proceed, please confirm:\n- " + "\n- ".join(qs))
    assistant_message = "\n\n".join([p for p in parts if p])

    # Create a conversation session and store initial messages
    conversation_id = new_conversation(run_id, result.get("route","other"), result.get("category","Other"), result.get("intent",""))
    add_message(conversation_id, "user", f"From: {email_in.sender_email}\nSubject: {subject}\n\n{body}", {"type":"email"})
    add_message(conversation_id, "assistant", assistant_message or "Processed.", {"type":"analysis"})
    print("Populating payload")
    payload = {
        "category": result.get("category","Other"),
        "route": result.get("route","other"),
        "intent": result.get("intent",""),
        "summary": result.get("summary",""),
        "recommendations": result.get("recommendations", []),
        "offers": result.get("offers", []),
        "drafted_email": result.get("drafted_email"),
        "crm_opportunity": result.get("crm_opportunity"),
        "customer_name": result.get("customer_name", ""),
        "product_name": result.get("product_name", ""),
        "purchase_order": result.get("purchaseOrderNumber", ""),
        "articleDoi": result.get("articleDoi", ""),
        "trace": trace,
        "conversation_id": conversation_id,
        "assistant_message": assistant_message,
    }
    print("Populated payload", payload)
    # Validate output schema (guardrail)
    out = ChatOut(**payload)
    return jsonify(out.model_dump())



@app.post("/api/conversation/message")
def conversation_message():
    data = request.get_json(force=True)
    conversation_id = data.get("conversation_id")
    msg = sanitize_text(data.get("message",""))
    if not conversation_id or not msg:
        return jsonify({"error":"conversation_id and message are required"}), 400

    # Load conversation context
    msgs = get_messages(conversation_id, limit=50)

    # Best-effort: reuse last run_id analysis from initial /api/chat stored in client
    # Client sends analysis/email snapshot for deterministic demo.
    analysis = data.get("analysis") or {}
    email = data.get("email") or {}

    add_message(conversation_id, "user", msg, {"type":"followup"})
    resp = respond(conversation_id, msg, analysis, email)
    add_message(conversation_id, "assistant", resp["message"], {"type":"assistant", "kb_hits": resp.get("kb_hits", [])})

    return jsonify({"conversation_id": conversation_id, "message": resp["message"], "kb_hits": resp.get("kb_hits", [])})


if __name__ == "__main__":
    # Flask debug can be toggled via env FLASK_DEBUG=1
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=bool(int(os.environ.get("FLASK_DEBUG","0"))))
