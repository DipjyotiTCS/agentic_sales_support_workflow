from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from db import conn_ctx
from llm import get_chat_model
from kb import retrieve_context
from audit import log_step
import json
import re
import uuid
from datetime import datetime, date
from query import query_kb
from answer_generator import generate_answer



support_intent_list = [
    "Access issue around product",
    "Refund request",
    "Technical issue",
    "Account verification",
    "Need more info from customer",
    "Other support"
]

sales_intent_list = [
    "Specific product related inquiry",
    "Customer requirement possible products",
    "Best price offer and bundling related query",
    "Order related query",
    "Need more info from customer",
    "Other sales"
]

# In LangGraph, when using `StateGraph(dict)`, node outputs may be treated as the
# new state (vs. partial updates) depending on version/config. To make the app
# robust across versions, every node returns the prior state with updates merged.
def _merged(state: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    new_state = dict(state)
    new_state.update(updates)
    return new_state

# --- helpers (SQL evidence) ---
def _sql_fetchone(sql: str,  *params):
    with conn_ctx() as conn:
        return conn.execute(sql, params).fetchone()

def _sql_fetchall(sql: str,  *params):
    with conn_ctx() as conn:
        return conn.execute(sql, params).fetchall()

# --- node: classification ---
def classify_node_1(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["email"]
    model = get_chat_model()

    if model is None:
        # Deterministic fallback
        text = (email["subject"] + " " + email["body"]).lower()
        if any(k in text for k in ["refund", "access", "login", "license", "error", "issue", "problem"]):
            category = "Support Type"
            conf = 0.78
        elif any(k in text for k in ["price", "quote", "discount", "buy", "purchase", "bundle", "offer", "order"]):
            category = "Sales Type"
            conf = 0.76
        else:
            category = "Other"
            conf = 0.55
        out = {"category": category, "confidence": conf}
        log_step(state["run_id"], "classify", email, out, conf, ["rule-based keywords"])
        return _merged(state, {"category": category, "category_confidence": conf})

    prompt = f"""
You are a classification agent for customer emails.

Return JSON with:
- category: one of ["Sales Type","Support Type","Other"]
- confidence: 0..1
- rationale: short

Email:
From: {email['sender_email']}
Subject: {email['subject']}
Body: {email['body']}
"""
    resp = model.invoke(prompt).content
    # Robust JSON extraction
    try:
        j = json.loads(resp[resp.find("{"):resp.rfind("}")+1])
    except Exception:
        j = {"category":"Other","confidence":0.4,"rationale":"Could not parse model output."}
    conf = float(j.get("confidence", 0.4))
    out = {"category": j.get("category","Other"), "confidence": conf, "rationale": j.get("rationale","")}
    log_step(state["run_id"], "classify", {"prompt":prompt}, out, conf, ["LLM JSON"])
    return _merged(state, {"category": out["category"], "category_confidence": conf})

# --- node: route ---
def route_node(state: Dict[str, Any]) -> Dict[str, Any]:
    cat = state.get("category","Other")
    route = "sales" if cat == "Sales Type" else "support" if cat == "Support Type" else "other"
    conf = 0.9 if route in ("sales","support") else 0.6
    log_step(state["run_id"], "route", {"category":cat}, {"route":route}, conf, [f"mapped {cat} -> {route}"])
    return _merged(state, {"route": route, "route_confidence": conf})

def support_route_node(state: Dict[str, Any]) -> Dict[str, Any]:
    route = state.get("intent","Other support")
    cat = state.get("category","Other")
    conf = 0.9 if route in ("Access issue around product","Refund request","Technical issue","Account verification","Need more info from customer","Other support") else 0.3
    rationale = f"Mapped from intent {route}"
    log_step(state["run_id"], "route", {"category":cat}, {"route":route, "rationale":rationale}, conf, [f"mapped {cat} -> {route}"])
    return _merged(state, {"route": route, "route_confidence": 1.0})

def sales_route_node(state: Dict[str, Any]) -> Dict[str, Any]:
    route = state.get("intent","Other sales")
    cat = state.get("category","Other")
    conf = 0.9 if route in ("Specific product related inquiry","Customer requirement possible products","Best price offer and bundling related query",
    "Order related query",
    "Need more info from customer",
    "Other sales") else 0.3
    rationale = f"Mapped from intent {route}"
    log_step(state["run_id"], "route", {"category":cat}, {"route":route, "rationale":rationale}, conf, [f"mapped {cat} -> {route}"])
    return _merged(state, {"route": route, "route_confidence": 1.0})

# --- node: kb retrieve (KB-first) ---
def kb_node(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["email"]
    query = f"{email['subject']}\n{email['body']}"
    ctx = retrieve_context(query, k=4)
    conf = 0.7 if ctx else 0.3
    log_step(state["run_id"], "kb_retrieve", {"query":query[:500]}, {"num_chunks":len(ctx)}, conf, ["top-k chunk retrieval"])
    return _merged(state, {"kb_context": ctx, "kb_confidence": conf})

# --- node: KB retrieve (sales/support) ---
def sales_kb_node(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["email"]
    query = f"[SALES]\nSubject: {email['subject']}\nBody: {email['body']}"
    ctx = retrieve_context(query, k=5)
    conf = 0.75 if ctx else 0.35
    log_step(state["run_id"], "sales_kb_retrieve", {"query": query[:500]}, {"num_chunks": len(ctx)}, conf, ["KB top-k retrieval (sales)"])
    return _merged(state, {"kb_context": ctx, "kb_confidence": conf})

def support_kb_node(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["email"]
    query = f"[SUPPORT]\nSubject: {email['subject']}\nBody: {email['body']}"

    hits = query_kb(query, top_k=5)
    answer = generate_answer(query, hits)

    conf = 0.75 if answer else 0.35
    log_step(state["run_id"], "support_kb_retrieve", {"query": query[:500]}, {"num_chunks": len(answer)}, conf, ["KB top-k retrieval (support)"])
    return _merged(state, {"kb_context": answer, "kb_confidence": conf})

# --- node: intent identification (must use KB findings in LLM call) ---
def sales_intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["email"]
    kb_ctx = state.get("kb_context", [])
    model = get_chat_model()
    allowed = [
        "Specific product related inquiry",
        "Customer requirement possible products",
        "Best price offer and bundling related query",
        "Order related query",
        "Need more info from customer",
        "Other sales"
    ]

    if model is None:
        txt = (email["subject"] + " " + email["body"]).lower()
        if "order" in txt and any(k in txt for k in ["where", "status", "track"]):
            intent, conf, rationale = "Order related query", 0.78, "Order tracking keywords detected."
        elif any(k in txt for k in ["price", "discount", "offer", "bundle", "bundling", "quote"]):
            intent, conf, rationale = "Best price offer and bundling related query", 0.76, "Pricing/bundling keywords detected."
        elif any(k in txt for k in ["recommend", "suggest", "which", "suitable", "need", "requirement"]):
            intent, conf, rationale = "Customer requirement possible products", 0.70, "Requirement/recommendation phrasing detected."
        elif any(k in txt for k in ["product", "feature", "spec", "model"]):
            intent, conf, rationale = "Specific product related inquiry", 0.68, "Product-specific keywords detected."
        else:
            intent, conf, rationale = "Other sales", 0.55, "Fallback."
        log_step(state["run_id"], "sales_intent", {"email": email, "kb_chunks": len(kb_ctx)}, {"intent": intent, "rationale": rationale}, conf, ["heuristic intent"])
        return _merged(state, {"intent": intent, "intent_confidence": conf, "intent_rationale": rationale})

    prompt = f"""
You are the SALES intent identification agent.

You MUST use the provided Knowledge Base findings to refine intent classification.
Return JSON with:
- intent: one of {allowed}
- confidence: 0..1
- rationale: short, explicitly referencing KB if relevant

Email:
Subject: {email['subject']}
Body: {email['body']}

Knowledge Base findings (top chunks):
{json.dumps(kb_ctx, indent=2)}
"""
    resp = model.invoke(prompt).content.strip()

    # Remove ```json ... ``` fences
    resp = re.sub(r"^```(?:json)?\s*", "", resp, flags=re.IGNORECASE)
    resp = re.sub(r"\s*```$", "", resp)

    try:
        data = json.loads(resp)
        intent = data.get("intent","Other sales")
        if intent not in allowed:
            intent = "Other sales"
        conf = float(data.get("confidence", 0.6))
        rationale = str(data.get("rationale",""))
    except Exception:
        intent, conf, rationale = "Other sales", 0.55, "Could not parse model output; defaulted."
    log_step(state["run_id"], "sales_intent", {"email": email, "kb_chunks": len(kb_ctx)}, {"intent": intent, "rationale": rationale}, conf, ["LLM + KB"])
    return _merged(state, {"intent": intent, "intent_confidence": conf, "intent_rationale": rationale})

def support_intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["email"]
    kb_ctx = state.get("kb_context", [])
    model = get_chat_model()
    allowed = [
        "Access issue around product",
        "Refund request",
        "Technical issue",
        "Account verification",
        "Need more info from customer",
        "Other support"
    ]

    if model is None:
        txt = (email["subject"] + " " + email["body"]).lower()
        if any(k in txt for k in ["refund", "return", "chargeback", "unused credits", "damaged"]):
            intent, conf, rationale = "Refund request", 0.78, "Refund-related keywords detected."
        elif any(k in txt for k in ["access", "login", "license", "subscription", "activate"]):
            intent, conf, rationale = "Access issue around product", 0.76, "Access/licensing keywords detected."
        elif any(k in txt for k in ["verify", "authenticate", "who am i", "identity"]):
            intent, conf, rationale = "Account verification", 0.68, "Verification keywords detected."
        elif any(k in txt for k in ["error", "issue", "problem", "bug", "not working"]):
            intent, conf, rationale = "Technical issue", 0.70, "Technical-problem keywords detected."
        else:
            intent, conf, rationale = "Other support", 0.55, "Fallback."
        log_step(state["run_id"], "support_intent", {"email": email, "kb_chunks": len(kb_ctx)}, {"intent": intent, "rationale": rationale}, conf, ["heuristic intent"])
        return _merged(state, {"intent": intent, "intent_confidence": conf, "intent_rationale": rationale})

    prompt = f"""
You are the SUPPORT intent identification agent.

You MUST use the provided Knowledge Base findings to refine intent classification.
Return JSON with:
- intent: one of {allowed}
- confidence: 0..1
- rationale: short, explicitly referencing KB if relevant

Email:
Subject: {email['subject']}
Body: {email['body']}

Knowledge Base findings (top chunks):
{json.dumps(kb_ctx, indent=2)}
"""
    resp = model.invoke(prompt).content.strip()

    # Remove ```json ... ``` fences
    resp = re.sub(r"^```(?:json)?\s*", "", resp, flags=re.IGNORECASE)
    resp = re.sub(r"\s*```$", "", resp)

    try:
        data = json.loads(resp)
        intent = data.get("intent","Other support")
        if intent not in allowed:
            intent = "Other support"
        conf = float(data.get("confidence", 0.6))
        rationale = str(data.get("rationale",""))
    except Exception:
        intent, conf, rationale = "Other support", 0.55, "Could not parse model output; defaulted."
    log_step(state["run_id"], "support_intent", {"email": email, "kb_chunks": len(kb_ctx)}, {"intent": intent, "rationale": rationale}, conf, ["LLM + KB"])
    return _merged(state, {"intent": intent, "intent_confidence": conf, "intent_rationale": rationale})


def classify_node(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["email"]
    model = get_chat_model()
    allowed = [
        "Access issue around product",
        "Refund request",
        "Technical issue",
        "Account verification",
        "Need more info from customer",
        "Other support",
        "Specific product related inquiry",
        "Customer requirement possible products",
        "Best price offer and bundling related query",
        "Order related query",
        "Need more info from customer",
        "Other sales"
    ]

    if model is None:
        txt = (email["subject"] + " " + email["body"]).lower()
        if any(k in txt for k in ["refund", "return", "chargeback", "unused credits", "damaged"]):
            intent, conf, rationale = "Refund request", 0.78, "Refund-related keywords detected."
        elif any(k in txt for k in ["access", "login", "license", "subscription", "activate"]):
            intent, conf, rationale = "Access issue around product", 0.76, "Access/licensing keywords detected."
        elif any(k in txt for k in ["verify", "authenticate", "who am i", "identity"]):
            intent, conf, rationale = "Account verification", 0.68, "Verification keywords detected."
        elif any(k in txt for k in ["error", "issue", "problem", "bug", "not working"]):
            intent, conf, rationale = "Technical issue", 0.70, "Technical-problem keywords detected."
        else:
            intent, conf, rationale = "Other support", 0.55, "Fallback."
        log_step(state["run_id"], "support_intent", {"email": email}, {"intent": intent, "rationale": rationale}, conf, ["heuristic intent"])
        return _merged(state, {"intent": intent, "intent_confidence": conf, "intent_rationale": rationale})

    prompt = f"""
You are the SUPPORT AND SALES intent identification agent.

You MUST use the provided email body and subject to classify the intent of the email.
Return JSON with:
- intent: one of {allowed}
- confidence: 0..1
- rationale: short and explicite reason to support your choice

Email:
Subject: {email['subject']}
Body: {email['body']}
"""
    resp = model.invoke(prompt).content.strip()
    category = "Others"

    # Remove ```json ... ``` fences
    resp = re.sub(r"^```(?:json)?\s*", "", resp, flags=re.IGNORECASE)
    resp = re.sub(r"\s*```$", "", resp)
    print("Raw LLM data", resp)
    try:
        data = json.loads(resp)
        intent = data.get("intent","General")
        if intent not in allowed:
            intent = "General"
        conf = float(data.get("confidence", 0.6))
        rationale = str(data.get("rationale",""))
    except Exception:
        intent, conf, rationale = "General", 0.55, "Could not parse model output; defaulted."

    if intent in sales_intent_list:
        category = "Sales Type"
    elif intent in support_intent_list:
        category = "Support Type"
       
    log_step(state["run_id"], "support_intent", {"email": email}, {"intent": intent, "rationale": rationale}, conf, ["LLM + KB"])
    return _merged(state, {"category": category, "intent": intent, "intent_confidence": conf, "category_confidence": conf, "intent_rationale": rationale})

# --- node: ticket log ---
def ticket_node(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["email"]
    route = state.get("route","other")
    # Insert into tickets if table exists (it does from CSV). Keep schema-agnostic: fill the common cols if present.
    evidence = []
    with conn_ctx() as conn:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(tickets)").fetchall()]
        # Build insert with available cols
        data = {}
        if "ticket_id" in cols:
            import uuid
            data["ticket_id"] = str(uuid.uuid4())
        if "ticket_number" in cols:
            data["ticket_number"] = f"TCK-{str(uuid.uuid4())[:8].upper()}"
        if "customer_email" in cols:
            data["customer_email"] = email["sender_email"]
        if "email_subject" in cols:
            data["email_subject"] = email["subject"]
        if "email_content" in cols:
            data["email_content"] = email["body"]
        if "category" in cols:
            data["category"] = "Sales" if route=="sales" else "Support" if route=="support" else "Other"
        if "status" in cols:
            data["status"] = "OPEN"
        if "priority" in cols:
            data["priority"] = "MEDIUM"
        if "created_at" in cols:
            import time
            data["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if "updated_at" in cols:
            import time
            data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        if data:
            keys = ", ".join([f'"{k}"' for k in data.keys()])
            qs = ", ".join(["?"]*len(data))
            conn.execute(f'INSERT INTO tickets ({keys}) VALUES ({qs})', tuple(data.values()))
            evidence.append(f"Inserted ticket with fields: {list(data.keys())}")
    log_step(state["run_id"], "ticket_log", {"route":route}, {"ticket_logged": bool(evidence)}, 0.85 if evidence else 0.4, evidence)
    return _merged(state, {"ticket_logged": True if evidence else False, "ticket_evidence": evidence})

# --- SALES flow ---
def sales_node(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["email"]
    text = (email["subject"] + " " + email["body"]).lower()

    # Use the intent identified by the sales_intent node (if present)
    force_intent = state.get("intent")

    # Identify customer tier
    cust = _sql_fetchone("SELECT * FROM customers WHERE email = ? LIMIT 1", (email["sender_email"],))
    tier = cust["tier"] if cust and "tier" in cust.keys() else "Standard"
    evidence = [f"Customer tier: {tier}"]

    # Intent detection (demo heuristics)
    if force_intent == "Order related query" or ("order" in text and ("where" in text or "status" in text or "track" in text)):
        intent = "Order related query"
        orders = _sql_fetchall("SELECT order_number, status, tracking_number, updated_at FROM orders WHERE customer_id = ?",
                               (cust["customer_id"],)) if cust else []
        if orders:
            top = orders[0]
            summary = f"Found order {top['order_number']} status={top['status']} tracking={top['tracking_number']}."
            evidence.append(f"orders: {len(orders)} rows")
        else:
            summary = "No order found for this customer in orders table."
            evidence.append("orders: 0 rows")
        conf = 0.8
        drafted = None

        log_step(state["run_id"], "sales_fulfillment", {"text":text[:300]}, {"intent":intent}, conf, evidence)
        return _merged(state, {"intent": intent, "summary": summary, "recommendations": [], "offers": [], "drafted_email": drafted, "sales_confidence": conf})

    if force_intent == "Best price offer and bundling related query" or any(k in text for k in ["price", "discount", "offer", "bundle", "bundling", "quote"]):
        intent = "Best price offer and bundling related query"
        # Fetch policies for tier
        policies = _sql_fetchall("SELECT * FROM pricing_policies WHERE customer_tier = ? AND active = 1", (tier,))
        max_disc = 10.0
        approval_thr = 15.0
        if policies:
            p = policies[0]
            max_disc = float(p["max_discount_percent"]) if "max_discount_percent" in p.keys() and p["max_discount_percent"] is not None else max_disc
            approval_thr = float(p["approval_threshold"]) if "approval_threshold" in p.keys() and p["approval_threshold"] is not None else approval_thr
            evidence.append(f"pricing_policy: {p['policy_name'] if 'policy_name' in p.keys() else 'n/a'} max_discount={max_disc} approval_threshold={approval_thr}")

        # Pick top 5 products by base_price descending "lucrative"
        prows = _sql_fetchall("SELECT product_id, name, base_price, category FROM products")
        prows = sorted(prows, key=lambda r: float(r["base_price"] or 0.0))
        top5 = prows[:5] if prows else []
        offers = []
        for r in top5:
            base = float(r["base_price"] or 0.0)
            # propose discount within max_disc
            disc = min(max_disc, 0.8 * max_disc)
            total = base * (1.0 - disc/100.0)
            compliant = disc <= max_disc and disc < approval_thr
            offers.append({
                "option_name": f"{r['name']} (bundle-ready)",
                "total_price": round(total, 2),
                "discount_percent": round(disc, 2),
                "compliant": bool(compliant),
                "evidence": [f"base_price={base}", f"tier={tier}", f"max_discount={max_disc}", f"approval_threshold={approval_thr}"],
                "reasoning": "Discount proposed within policy limits; if approval needed, route to human approval."
            })
        summary = "Prepared top pricing/bundling options honoring discount policy."
        conf = 0.77
        log_step(state["run_id"], "sales_pricing", {"tier":tier}, {"offers":len(offers)}, conf, evidence)
        return _merged(state, {"intent": intent, "summary": summary, "recommendations": [], "offers": offers, "drafted_email": None, "sales_confidence": conf})

    # product inquiry / requirements → recommend products (simple keyword match on category/description)
    intent = "Specific product related inquiry"
    prows = _sql_fetchall("SELECT product_id, name, description, category FROM products")
    q = set(text.split())
    scored = []
    for r in prows:
        words = set((str(r["name"]) + " " + str(r["description"]) + " " + str(r["category"])).lower().split())
        score = len(q.intersection(words)) / max(1, len(q))
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    recs = []
    for s, r in scored[:5]:
        recs.append({
            "product_id": r["product_id"],
            "name": r["name"],
            "purpose": f"Matches category {r['category']}",
            "score": float(min(1.0, s*3)),
            "reasoning": "Keyword match between email and product metadata; refine with KB/LLM when available."
        })

    summary = "Recommended products based on customer ask and product catalog evidence."
    conf = 0.7 if recs else 0.5
    drafted = None
    if "need" in text or "requirement" in text:
        drafted = f"""Subject: Re: {email['subject']}

Hi,

Thanks for reaching out. To recommend the best-fit options, could you share:
1) Target use case / key requirements (top 3)
2) Expected users / seats
3) Desired contract duration
4) Budget range (if any)
5) Must-have integrations / compliance needs

Once I have this, I will propose the most suitable product(s) with pricing and bundling options.

Regards,
Sales Team
"""

    log_step(state["run_id"], "sales_recommend", {"email":email}, {"recs":len(recs)}, conf, evidence)
    return _merged(state, {"intent": intent, "summary": summary, "recommendations": recs, "offers": [], "drafted_email": drafted, "sales_confidence": conf})

# --- SUPPORT flow ---
def support_node(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["email"]
    text = (email["subject"] + " " + email["body"]).lower()

    # Auth check: is sender in customers table?
    cust = _sql_fetchone("SELECT * FROM customers WHERE email = ? LIMIT 1", (email["sender_email"],))
    evidence = []
    if not cust:
        intent = "User authentication failed"
        summary = "Sender email not found in customer registry. Request additional verification."
        drafted = f"""Subject: Re: {email['subject']}

Hi,

For security, we couldn't validate your account with the email address used. Please reply with:
- Registered email / customer ID
- Company name
- Last invoice / order number (if available)

Once verified, we will proceed with your support request.

Regards,
Support Team
"""
        conf = 0.85
        log_step(state["run_id"], "support_auth", {"sender":email["sender_email"]}, {"authenticated":False}, conf, ["customers lookup: not found"])
        return _merged(state, {"intent": intent, "summary": summary, "drafted_email": drafted, "support_confidence": conf, "recommendations": [], "offers": []})

    evidence.append("customers lookup: found")
    # Access/licensing
    if any(k in text for k in ["access", "login", "license", "licence", "subscription"]):
        intent = "Access issue around product"
        subs = _sql_fetchall("SELECT * FROM subscriptions WHERE customer_id = ? AND status = 'ACTIVE'", (cust["customer_id"],))
        if not subs:
            summary = "No active subscription found. Access cannot be granted; route to sales for renewal."
            conf = 0.8
            evidence.append("subscriptions: 0 active")
        else:
            s0 = subs[0]
            prod = _sql_fetchone("SELECT * FROM products WHERE product_id = ? LIMIT 1", (s0["product_id"],))
            summary = f"Active subscription found for product {prod['name'] if prod else s0['product_id']}. Verify license period and counts (demo: counts not provided in CSV)."
            conf = 0.78
            evidence.append(f"subscriptions: {len(subs)} active")
        log_step(state["run_id"], "support_access", {"customer_id":cust["customer_id"]}, {"subs_active":len(subs)}, conf, evidence)
        return _merged(state, {"intent": intent, "summary": summary, "drafted_email": None, "support_confidence": conf, "recommendations": [], "offers": []})

    # Refund flow
    if "refund" in text or "unused credit" in text or "damaged" in text:
        intent = "Refund request"
        # Try to infer product via keyword match
        prows = _sql_fetchall("SELECT product_id, name FROM products")
        chosen = None
        for r in prows:
            if str(r["name"]).lower() in text:
                chosen = r
                break
        if chosen:
            pol = _sql_fetchone("SELECT * FROM refund_policies WHERE product_id = ? AND active = 1 LIMIT 1", (chosen["product_id"],))
            if pol:
                needs_approval = bool(pol["requires_approval"]) if "requires_approval" in pol.keys() else True
                pct = float(pol["refund_percentage"] or 0.0) if "refund_percentage" in pol.keys() else 0.0
                summary = f"Refund policy found for {chosen['name']}: refund_percentage={pct}%. Requires approval={needs_approval}. Prepared evidence for human authorization."
                conf = 0.76
                evidence += [f"refund_policy_id={pol['policy_id']}", f"refund_percentage={pct}", f"requires_approval={needs_approval}"]
            else:
                summary = f"No active refund policy found for {chosen['name']}. Likely not eligible; provide evidence to sales rep."
                conf = 0.7
                evidence += ["refund_policies: none"]
        else:
            summary = "Could not identify product for refund. Ask customer for product/order details."
            conf = 0.65
            evidence += ["product inference: none"]
        log_step(state["run_id"], "support_refund", {"text":text[:400]}, {"product": chosen["product_id"] if chosen else None}, conf, evidence)
        return _merged(state, {"intent": intent, "summary": summary, "drafted_email": None, "support_confidence": conf, "recommendations": [], "offers": []})

    intent = "General support query"
    summary = "Logged support ticket. Need more details to proceed."
    drafted = f"""Subject: Re: {email['subject']}

Hi,

Thanks for contacting support. To help quickly, please share:
- Product name
- Steps to reproduce / exact error message
- Screenshot (if available)
- Your environment (OS/browser/app version)

Regards,
Support Team
"""
    conf = 0.6
    log_step(state["run_id"], "support_general", {"text":text[:300]}, {"intent":intent}, conf, evidence)
    return _merged(state, {"intent": intent, "summary": summary, "drafted_email": drafted, "support_confidence": conf, "recommendations": [], "offers": []})

# --- OTHER flow ---
def other_node(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["email"]
    summary = "Email does not match sales/support. Logged for manual triage."
    intent = "Other"
    conf = 0.55
    log_step(state["run_id"], "other", {"subject":email["subject"]}, {"intent":intent}, conf, ["default bucket"])
    return _merged(state, {"intent": intent, "summary": summary, "drafted_email": None, "recommendations": [], "offers": [], "other_confidence": conf})


# --- end-of-flow nodes (shared) ---
def finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize: consolidate key fields and ensure required outputs exist."""
    # Compute a simple average confidence across known confidence fields
    conf_fields = [
        state.get("category_confidence"),
        state.get("route_confidence"),
        state.get("kb_confidence"),
        state.get("intent_confidence"),
        state.get("sales_confidence"),
        state.get("support_confidence"),
        state.get("other_confidence"),
    ]
    vals = [float(x) for x in conf_fields if isinstance(x, (int, float))]
    avg = sum(vals) / len(vals) if vals else 0.0
    log_step(state["run_id"], "finalize", {"route": state.get("route")}, {"avg_confidence": avg}, 0.9, ["consolidated outputs"])
    return _merged(state, {"avg_confidence": avg})


def present_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Present: last node before END. Keeps state unchanged but logs completion."""
    log_step(state["run_id"], "present", {"route": state.get("route")}, {"status": "done"}, 0.95, ["ready to present"])
    return state


def unknown_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Unknown: for non sales/support; uses existing other_node behavior."""
    return other_node(state)

def build_refund_graph():
    refund = StateGraph(dict)
    refund.add_node("extract_details", extract_details_node)
    refund.add_node("refund_validate", refund_validation_node)
    refund.add_node("refund_case_creation", create_refund_case_node)
    refund.add_node("refund_calculate", calculate_refund_node)
    refund.add_node("refund_info", refund_info_node)

    refund.set_entry_point("extract_details")

    refund.add_edge("extract_details", "refund_validate")
    refund.add_edge("refund_validate", "refund_case_creation")
    refund.add_edge("refund_case_creation", "refund_calculate")
    refund.add_edge("refund_calculate", "refund_info")
    refund.add_edge("refund_info", END)
    return refund.compile()

def extract_details_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("coming to extract the details")
    email = state["email"]
    model = get_chat_model()

    if model is None:
        txt = (email["subject"] + " " + email["body"]).lower()
        if any(k in txt for k in ["refund", "return", "chargeback", "unused credits", "damaged"]):
            intent, conf, rationale = "Refund request", 0.78, "Refund-related keywords detected."
        elif any(k in txt for k in ["access", "login", "license", "subscription", "activate"]):
            intent, conf, rationale = "Access issue around product", 0.76, "Access/licensing keywords detected."
        elif any(k in txt for k in ["verify", "authenticate", "who am i", "identity"]):
            intent, conf, rationale = "Account verification", 0.68, "Verification keywords detected."
        elif any(k in txt for k in ["error", "issue", "problem", "bug", "not working"]):
            intent, conf, rationale = "Technical issue", 0.70, "Technical-problem keywords detected."
        else:
            intent, conf, rationale = "Other support", 0.55, "Fallback."
        log_step(state["run_id"], "support_intent", {"email": email}, {"intent": intent, "rationale": rationale}, conf, ["heuristic intent"])
        return _merged(state, {"intent": intent, "intent_confidence": conf, "intent_rationale": rationale})

    prompt = f"""
You are the best refund request email ANALYSIS agent.

You MUST analyse the provided email body and subject to identify customer's email id, Purchase Order Number, Article DOI and Reason for Refund
Return JSON with:
- customerEmailId: identified customer email id
- purchaseOrderNumber: identified Purchase Order Number
- articleDoi: identified Article DOI
- refundReason: identified Reason for Refund
- confidence: 0..1
- rationale: short and explicite reason to support your choice
In case you are not able to find out any of the above mentioned information fill the respective JSON field with text unidentified.

Email:
Subject: {email['subject']}
Body: {email['body']}
"""
    resp = model.invoke(prompt).content.strip()

    # Remove ```json ... ``` fences
    resp = re.sub(r"^```(?:json)?\s*", "", resp, flags=re.IGNORECASE)
    resp = re.sub(r"\s*```$", "", resp)
    print("Raw LLM data", resp)
    try:
        data = json.loads(resp)
        customerEmailId = data.get("customerEmailId", email)

        if not customerEmailId or customerEmailId.strip().lower() in {"unidentified", "unknown", "n/a"}:
            customerEmailId = email['sender_email']
        
        purchaseOrderNumber = data.get("purchaseOrderNumber","unidentified")
        articleDoi = data.get("articleDoi","unidentified")
        refundReason = data.get("refundReason","unidentified")
        refund_conf = float(data.get("confidence", 0.6))
        refund_rationale = data.get("rationale","unidentified")
        print("Extracted values", refund_rationale)
    except Exception:
        intent, refund_conf, rationale = "General", 0.3, "Could not parse model output; defaulted."

    log_step(state["run_id"], "data_extraction", {"email": email}, {"refund_conf": refund_conf, "rationale": refund_rationale}, refund_conf, ["LLM + DB"])
    return _merged(state, {"customerEmailId": customerEmailId, "purchaseOrderNumber": purchaseOrderNumber, "articleDoi": articleDoi, "refundReason": refundReason, "refund_conf": refund_conf, "refund_rationale": refund_rationale})

def refund_validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    is_refundable = False
    try:
        customerEmailId = state['customerEmailId']
        purchaseOrderNumber = state['purchaseOrderNumber']
        cust = _sql_fetchone("SELECT * FROM customers WHERE email = ? LIMIT 1", customerEmailId)
        customer_id = cust["customer_id"]
        customer_name = cust["name"]
        order = _sql_fetchone("SELECT * FROM orders WHERE order_number = ? AND customer_id = ? LIMIT 1", purchaseOrderNumber, customer_id)
        product_id = order["product_id"]
        order_status = order["status"]
        order_amount = order["total_amount"]
        if order_status != "cancelled":
            product = _sql_fetchone("SELECT * FROM products WHERE product_id = ? LIMIT 1", product_id)
            product_name = product["name"]
            is_refundable = product["is_refundable"]
            validation_result = f"Order found with status {order_status}. Product refundable: {is_refundable}."
        log_step(state["run_id"], "user_authetication", {"email": customerEmailId}, {"purchaseOrderNumber": purchaseOrderNumber, "customer_id": customer_id, "product_id": product_id, "is_refundable": is_refundable, "order_amount": order_amount, "rationale": validation_result}, 1.0, ["Calculated from DB"])
        return _merged(state, {"customerEmailId": customerEmailId, "purchaseOrderNumber": purchaseOrderNumber, "customer_id": customer_id, "product_id": product_id, "is_refundable": is_refundable, "order_amount": order_amount, "customer_name": customer_name, "product_name": product_name})
    except Exception as e:
        print("Exception occured", e)
        customerEmailId, customer_id, product_id = "General", 0.3, "Could not parse model output; defaulted."
        validation_result = "User authetication failed. Could not validate refund eligibility due to missing or invalid data."
        order_amount = 0.0
        product_name = "unknown"
        log_step(state["run_id"], "user_authetication", {"email": customerEmailId}, {"purchaseOrderNumber": purchaseOrderNumber, "customer_id": customer_id, "product_id": product_id, "is_refundable": is_refundable, "order_amount": order_amount, "rationale": validation_result}, 0.3, ["Calculated from DB"])
        return _merged(state, {"customerEmailId": customerEmailId, "purchaseOrderNumber": purchaseOrderNumber, "customer_id": customer_id, "product_id": product_id, "is_refundable": is_refundable, "order_amount": order_amount})

def create_refund_case_node(state: Dict[str, Any]) -> Dict[str, Any]:
    #Create a case with OSC
    customerEmailId = state['customerEmailId']
    is_refundable = state['is_refundable']
    print("Value of is refundable", is_refundable)
    if is_refundable:
        unique_id = uuid.uuid4().hex
        case_node_info = f"Created refund case with OSC. Case ID: {unique_id} for email {customerEmailId}."
        log_step(state["run_id"], "OSC_case_creation", {"email": customerEmailId}, {"oracle_service_ticket_number": unique_id, "rationale": case_node_info}, 1.0, ["Created via OSC"])
        return _merged(state, {"oracle_service_ticket_number": unique_id})

def calculate_refund_node(state: Dict[str, Any]) -> Dict[str, Any]:
    is_refundable = state['is_refundable']
    customerEmailId = state['customerEmailId']
    customer_id = state['customer_id']
    product_id = state['product_id']
    purchaseOrderNumber = state['purchaseOrderNumber']
    if is_refundable:
        try:
            total_price = state['order_amount']
            order = _sql_fetchone("SELECT * FROM orders WHERE order_number = ? AND customer_id = ? LIMIT 1", purchaseOrderNumber, customer_id)
            order_created_at = order["created_at"]
            policies = _sql_fetchone("SELECT * FROM refund_policies WHERE product_id = ? LIMIT 1", product_id)
            refund_window_days = policies["refund_window_days"]
            print("Refund window", refund_window_days)
            refund_percentage = policies["refund_percentage"]
            order_date = datetime.strptime(order_created_at, "%Y-%m-%dT%H:%M:%S.%f").date()
            today = date.today()
            date_diff = (today - order_date).days
            print("Date difference", date_diff)
            if refund_window_days >= date_diff:
                refund_amount = (total_price * refund_percentage)/100
                message = "Allowed refund amount is " + str(refund_amount)
            else:
               is_refundable = False
               refund_amount = 0.0
               message = "Refund cannot be processed as refund window has passed."     
            log_step(state["run_id"], "refund_calculation", {"email": customerEmailId}, {"refund_amount": refund_amount, "rationale": message}, 1.0, ["Calculated via DB"])   
        except Exception as e:
            print("Exception occured", e)
            purchaseOrderNumber, customer_id, product_id = "General", 0.55, "Could not parse model output; defaulted."
            log_step(state["run_id"], "OSC_case_creation", {"email": customerEmailId}, {"refund_amount": 0.0, "rationale": "Could not process refund. Please check manually."}, 0.3, ["Calculated via DB"])
        return _merged(state, {"is_refundable": is_refundable, "refund_amount": refund_amount, "message": message})

def refund_info_node(state: Dict[str, Any]) -> Dict[str, Any]:
    is_refundable = state['is_refundable']
    customerEmailId = state['customerEmailId']
    if is_refundable:
        info_message = "Notification email sent to example@example.com for refund approval."
        log_step(state["run_id"], "refund_info_node", {"email": customerEmailId}, {"rationale": info_message}, 1.0, ["LLM"])   
    return _merged(state, {"avg_confidence": "Hello"})

def build_access_graph():
    rg = StateGraph(dict)
    return rg.compile

def build_technical_graph():
    rg = StateGraph(dict)
    return rg.compile

def build_account_graph():
    rg = StateGraph(dict)
    return rg.compile

def build_more_info_graph():
    rg = StateGraph(dict)
    return rg.compile

def build_other_graph():
    rg = StateGraph(dict)
    return rg.compile

def build_product_inquiry_graph():
    product_inquiry = StateGraph(dict)
    product_inquiry.add_node("support_kb_node", support_kb_node)
    product_inquiry.add_node("formulate_response", formulate_response)

    product_inquiry.set_entry_point("support_kb_node")

    product_inquiry.add_edge("support_kb_node", "formulate_response")
    product_inquiry.add_edge("formulate_response", END)
    return product_inquiry.compile()

def formulate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    return _merged(state, {"kb_context": "ctx", "kb_confidence": "conf"})

def build_customer_requirement_graph():
    rg = StateGraph(dict)
    return rg.compile

def build_best_price_graph():
    rg = StateGraph(dict)
    return rg.compile

def build_order_quiry_graph():
    rg = StateGraph(dict)
    return rg.compile

def build_other_sales_graph():
    rg = StateGraph(dict)
    return rg.compile

def build_sales_graph():
    """Sales subgraph: KB -> intent -> ticket -> fulfillment."""
    product_inquiry_graph = build_product_inquiry_graph()
    customer_requirement_graph = build_customer_requirement_graph()
    best_price_graph = build_best_price_graph()
    order_quiry_graph = build_order_quiry_graph()
    more_info_graph = build_more_info_graph()
    other_sales_graph = build_other_sales_graph()
    sg = StateGraph(dict)
    sg.add_node("route", sales_route_node)

    # Subgraphs as nodes
    sg.add_node("product_inquiry", product_inquiry_graph)
    sg.add_node("customer_requirement", customer_requirement_graph)
    sg.add_node("best_price", best_price_graph)
    sg.add_node("order_quiry", order_quiry_graph)
    sg.add_node("moreinfo", more_info_graph)
    sg.add_node("other", other_sales_graph)

    sg.set_entry_point("route")

    # Select which subgraph to run based on support intent-derived route
    def _after_route(state: Dict[str, Any]) -> str:
        return state.get("route", "Other sales")

    sg.add_conditional_edges(
        "route",
        _after_route,
        {"Specific product related inquiry": "product_inquiry", "Customer requirement possible products": "customer_requirement", 
         "Best price offer and bundling related query": "best_price", "Order related query": "order_quiry",
        "Need more info from customer": "moreinfo", "Other sales": "other"},
    )

    sg.add_edge("route", END)
    return sg.compile()

def build_support_graph():
    """Support subgraph: KB -> intent -> ticket -> fulfillment."""
    refund_graph = build_refund_graph()
    access_graph = build_access_graph()
    technical_graph = build_technical_graph()
    account_graph = build_account_graph()
    more_info_graph = build_more_info_graph()
    other_graph = build_other_graph()
    sg = StateGraph(dict)
    sg.add_node("route", support_route_node)

    # Subgraphs as nodes
    sg.add_node("refund", refund_graph)
    sg.add_node("access", access_graph)
    sg.add_node("technical", technical_graph)
    sg.add_node("account", account_graph)
    sg.add_node("moreinfo", more_info_graph)
    sg.add_node("other", other_graph)

    sg.set_entry_point("route")

    # Select which subgraph to run based on support intent-derived route
    def _after_route(state: Dict[str, Any]) -> str:
        return state.get("route", "Other support")

    sg.add_conditional_edges(
        "route",
        _after_route,
        {"Access issue around product": "access", "Refund request": "refund", "Technical issue": "technical", 
         "Account verification": "account", "Need more info from customer": "moreinfo", "Other support": "other"},
    )

    sg.add_edge("route", END)
    return sg.compile()

def build_support_graph_old():
    """Support subgraph: KB -> intent -> ticket -> fulfillment."""
    sg = StateGraph(dict)
    sg.add_node("support_kb", support_kb_node)
    sg.add_node("ticket", ticket_node)
    sg.add_node("support", support_node)
    sg.set_entry_point("support_kb")
    sg.add_edge("support_kb", "ticket")
    sg.add_edge("ticket", "support")
    sg.add_edge("support", END)
    return sg.compile()

def build_graph():
    # Build subgraphs
    sales_graph = build_sales_graph()
    support_graph = build_support_graph()

    g = StateGraph(dict)
    g.add_node("classify", classify_node)
    g.add_node("route", route_node)

    # Subgraphs as nodes
    g.add_node("sales", sales_graph)
    g.add_node("support", support_graph)

    # Unknown + shared tail
    g.add_node("unknown", unknown_node)
    g.add_node("finalize", finalize_node)
    g.add_node("present", present_node)

    # ENTRY
    g.set_entry_point("classify")
    g.add_edge("classify", "route")

    # Select which subgraph to run based on classification-derived route
    def _after_route(state: Dict[str, Any]) -> str:
        return state.get("route", "unknown")

    g.add_conditional_edges(
        "route",
        _after_route,
        {"sales": "sales", "support": "support", "other": "unknown", "unknown": "unknown"},
    )

    # After sales/support/unknown: finalize and present
    g.add_edge("sales", "finalize")
    g.add_edge("support", "finalize")
    g.add_edge("unknown", "finalize")
    g.add_edge("finalize", "present")
    g.add_edge("present", END)
    return g.compile()
