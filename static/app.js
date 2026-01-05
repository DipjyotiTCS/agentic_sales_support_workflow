async function postJSON(url, obj) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(obj)
  });
  const txt = await res.text();
  let data;
  try { data = JSON.parse(txt); } catch(e) { data = {raw: txt}; }
  if (!res.ok) throw new Error(data.error || data.message || txt);
  return data;
}

function fmtPct(x){
  if (x === null || x === undefined || isNaN(x)) return "—";
  return Math.round(x * 100) + "%";
}

function toast(msg){
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.style.display = "block";
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => { el.style.display = "none"; }, 4200);
}

function setStatus(id, msg){
  const el = document.getElementById(id);
  if (el) el.textContent = msg || "";
}

function renderKPI(data){
  const kpi = document.getElementById("kpi");
  kpi.innerHTML = "";
  const items = [
    ["Category", data.category || "—"],
    ["Route", data.route || "—"],
    ["Intent", data.intent || "—"],
  ];
  if (Array.isArray(data.trace) && data.trace.length){
    const avg = data.trace.reduce((a,s)=>a+(s.confidence||0),0) / data.trace.length;
    items.push(["Avg confidence", fmtPct(avg)]);
  }
  for (const [k,v] of items){
    const pill = document.createElement("div");
    pill.className = "pill";
    pill.innerHTML = `<span>${k}:</span> <strong>${escapeHtml(String(v))}</strong>`;
    kpi.appendChild(pill);
  }
}

function escapeHtml(s){
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
          .replace(/"/g,"&quot;").replace(/'/g,"&#039;");
}

function renderSummary(data){
  const lines = [];
  lines.push(`Summary:\n${data.summary || "—"}`);
  if (data.recommendations && data.recommendations.length){
    lines.push("\nRecommendations:");
    data.recommendations.forEach(r => {
      lines.push(`- ${r.name} (${fmtPct(r.score)}): ${r.purpose}\n  Reason: ${r.reasoning}`);
    });
  }
  if (data.offers && data.offers.length){
    lines.push("\nOffers (Top 5):");
    data.offers.forEach(o => {
      lines.push(`- ${o.option_name}: total=${o.total_price} discount%=${o.discount_percent} compliant=${o.compliant}`);
    });
  }
  if (data.drafted_email){
    lines.push("\nDrafted Email:");
    lines.push(data.drafted_email);
  }
  if (data.crm_opportunity){
    lines.push("\nCRM Opportunity:");
    lines.push(JSON.stringify(data.crm_opportunity, null, 2));
  }
  document.getElementById("summaryBox").textContent = lines.join("\n");
  document.getElementById("rawBox").textContent = JSON.stringify(data, null, 2);
}

function renderTrace(data){
  const box = document.getElementById("traceBox");
  box.innerHTML = "";
  const steps = Array.isArray(data.trace) ? data.trace : [];
  if (!steps.length){
    box.innerHTML = `<div class="small">No trace available.</div>`;
    return;
  }

  steps.forEach((s) => {
    const step = document.createElement("div");
    step.className = "step";

    const conf = (typeof s.confidence === "number") ? s.confidence : 0;
    const pct = Math.max(0, Math.min(100, Math.round(conf * 100)));

    const ev = Array.isArray(s.evidence) ? s.evidence : [];
    const evHtml = ev.length
      ? `<div class="evidence"><div><strong>Evidence</strong></div><ul>${ev.map(e=>`<li>${escapeHtml(String(e))}</li>`).join("")}</ul></div>`
      : "";

    step.innerHTML = `
      <div class="stepHead">
        <div class="stepName">${escapeHtml(String(s.step || "step"))}</div>
        <div class="conf">
          <div class="bar"><span style="width:${pct}%"></span></div>
          <div class="badge">${pct}%</div>
        </div>
      </div>
      <div class="reason">${escapeHtml(String(s.reasoning || ""))}</div>
      ${evHtml}
    `;
    box.appendChild(step);
  });
}

function wireTabs(){
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach(t => {
    t.addEventListener("click", () => {
      tabs.forEach(x => x.classList.remove("active"));
      t.classList.add("active");
      const key = t.getAttribute("data-tab");
      document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
      document.getElementById(`panel-${key}`).classList.add("active");
    });
  });
}

async function uploadKB(){
  const file = document.getElementById("kbFile").files[0];
  if (!file){
    toast("Please choose a PDF first.");
    return;
  }
  setStatus("kbStatus", "Uploading and processing...");
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/kb/upload", { method:"POST", body: fd });
  const txt = await res.text();
  let data;
  try { data = JSON.parse(txt); } catch(e) { data = { raw: txt }; }
  if (!res.ok){
    setStatus("kbStatus", "");
    throw new Error(data.error || data.message || txt);
  }
  setStatus("kbStatus", `Ingested: ${data.filename} • pages=${data.pages} • chunks=${data.chunks} • embedded=${data.embedded}`);
  toast("KB updated.");
}

async function runChat(){
  const sender = document.getElementById("sender").value.trim();
  const subject = document.getElementById("subject").value.trim();
  const body = document.getElementById("body").value.trim();

  setStatus("status", "Running agent graph...");
  document.getElementById("runBtn").disabled = true;

  try{    const data = await postJSON("/api/chat", { sender_email: sender, subject, body });

    // Conversation panel
    __conversationId = data.conversation_id || null;
    __lastAnalysis = data || null;
    __lastEmail = { sender_email: sender, subject, body };
    const thread = document.getElementById("chatThread");
    if (thread){ thread.innerHTML = ""; }
    appendBubble("user", `From: ${sender}
Subject: ${subject}

${body}`);
    appendBubble("assistant", data.assistant_message || data.summary || "Processed.");

    renderKPI(data);
    renderSummary(data);
    renderTrace(data);
toast("Run completed.");
    setStatus("status", "");
  }catch(e){
    console.error(e);
    toast(e.message || String(e));
    setStatus("status", "Error: " + (e.message || String(e)));
  }finally{
    document.getElementById("runBtn").disabled = false;
  }
}

function resetForm(){
  document.getElementById("sender").value = "";
  document.getElementById("subject").value = "";
  document.getElementById("body").value = "";
  document.getElementById("kpi").innerHTML = "";
  document.getElementById("summaryBox").textContent = "Run an email to see results.";
  document.getElementById("traceBox").innerHTML = "";
  document.getElementById("rawBox").textContent = "";
  setStatus("status","");
}

document.addEventListener("DOMContentLoaded", () => {
  wireTabs();
  document.getElementById("runBtn").addEventListener("click", runChat);
  document.getElementById("resetBtn").addEventListener("click", resetForm);
  document.getElementById("kbBtn").addEventListener("click", () => uploadKB().catch(e => {
    console.error(e);
    toast(e.message || String(e));
    setStatus("kbStatus", "Error: " + (e.message || String(e)));
  }));
});


let __conversationId = null;
let __lastAnalysis = null;
let __lastEmail = null;

function appendBubble(role, text){
  const thread = document.getElementById("chatThread");
  if (!thread) return;
  const b = document.createElement("div");
  b.className = "bubble " + (role === "user" ? "user" : "assistant");
  b.textContent = text || "";
  thread.appendChild(b);
  thread.scrollTop = thread.scrollHeight;
}

function resetConversation(){
  const thread = document.getElementById("chatThread");
  if (thread) thread.innerHTML = "";
  __conversationId = null;
  __lastAnalysis = null;
  __lastEmail = null;
  appendBubble("assistant", "Paste an email above and click “Analyze”. I’ll respond here, and you can continue the conversation.");
}

async function sendFollowup(){
  const inp = document.getElementById("followupInput");
  const msg = (inp?.value || "").trim();
  if (!msg) return;
  if (!__conversationId){
    toast("Run an email first to start a conversation.");
    return;
  }
  appendBubble("user", msg);
  inp.value = "";
  try{
    const data = await postJSON("/api/conversation/message", {
      conversation_id: __conversationId,
      message: msg,
      analysis: __lastAnalysis || {},
      email: __lastEmail || {}
    });
    appendBubble("assistant", data.message || "");
  }catch(e){
    toast(e.message || String(e));
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("sendFollowupBtn");
  if (btn) btn.addEventListener("click", sendFollowup);
  const inp = document.getElementById("followupInput");
  if (inp) inp.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") sendFollowup();
  });
  resetConversation();
});
