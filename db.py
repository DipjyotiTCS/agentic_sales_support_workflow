import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.environ.get("APP_DB_PATH", os.path.join(os.path.dirname(__file__), "app.db"))

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def conn_ctx():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def exec_script(sql: str) -> None:
    with conn_ctx() as conn:
        conn.executescript(sql)
