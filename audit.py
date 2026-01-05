import json, time, uuid
from typing import Any, Dict, List, Optional
from db import conn_ctx

def ensure_audit_tables():
    with conn_ctx() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_runs(
            run_id TEXT,
            created_at TEXT,
            step_name TEXT,
            input_json TEXT,
            output_json TEXT,
            confidence REAL,
            evidence_json TEXT
        );
        """)

def new_run_id() -> str:
    return str(uuid.uuid4())

def log_step(run_id: str, step_name: str, inp: Any, out: Any, confidence: float, evidence: List[str]):
    ensure_audit_tables()
    with conn_ctx() as conn:
        conn.execute(
            "INSERT INTO agent_runs(run_id, created_at, step_name, input_json, output_json, confidence, evidence_json) VALUES(?,?,?,?,?,?,?)",
            (
                run_id,
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                step_name,
                json.dumps(inp, default=str),
                json.dumps(out, default=str),
                float(confidence),
                json.dumps(evidence, default=str),
            ),
        )
