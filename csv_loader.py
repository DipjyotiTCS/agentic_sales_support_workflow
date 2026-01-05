import os
import pandas as pd
from typing import Dict, Tuple
from db import conn_ctx

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

CSV_TABLES = {
    "customers": "customers.csv",
    "products": "products.csv",
    "subscriptions": "subscriptions.csv",
    "orders": "orders.csv",
    "pricing_policies": "pricing_policies.csv",
    "refund_policies": "refund_policies.csv",
    "tickets": "tickets.csv",
}

def _infer_sql_type(series: pd.Series) -> str:
    # Keep it simple for demo: TEXT for everything except obvious numerics/bools
    if pd.api.types.is_bool_dtype(series):
        return "INTEGER"
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"

def ensure_tables_from_csv() -> Dict[str, Tuple[int,int]]:
    """
    Creates tables based on CSV headers and loads rows at startup.
    Returns {table: (rows, cols)}.
    """
    stats = {}
    with conn_ctx() as conn:
        for table, fname in CSV_TABLES.items():
            path = os.path.join(DATA_DIR, fname)
            df = pd.read_csv(path)
            # Normalize NaNs to None so sqlite stores NULL
            df = df.where(pd.notnull(df), None)

            cols = list(df.columns)
            types = {c: _infer_sql_type(df[c]) for c in cols}

            col_defs = ", ".join([f'"{c}" {types[c]}' for c in cols])
            conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({col_defs});')
            # Clear then load for deterministic demo
            conn.execute(f'DELETE FROM "{table}";')

            placeholders = ", ".join(["?"] * len(cols))
            col_names = ", ".join([f'"{c}"' for c in cols])
            insert_sql = f'INSERT INTO "{table}" ({col_names}) VALUES ({placeholders});'
            conn.executemany(insert_sql, df.itertuples(index=False, name=None))

            stats[table] = (len(df), len(cols))
    return stats
