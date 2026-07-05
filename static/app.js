const state = { startup: "", agent: "finance", hrDatasets: [], legalCats: [], llm: false };

const $ = (s) => document.querySelector(s);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const money = (n) => "$" + Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 });
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

const AGENTS = {
  finance: {
    title: "Finance Agent",
    desc: "Ask about your company's financial health.",
    placeholder: "Ask about cash, burn, runway, profit\u2026",
    es: ["\uD83D\uDCB0", "Finance Agent", "Pick a preset below or type any financial question."],
  },
  hr: {
    title: "Hiring Agent",
    desc: "Bias-aware candidate shortlisting.",
    placeholder: "e.g. shortlist candidates from the income dataset\u2026",
    es: ["\uD83E\uDDD1\u200D\uD83D\uDCBC", "Hiring Agent", "Connect a dataset below and run a fairness-audited shortlist powered by CatBoost."],
  },
  legal: {
    title: "Legal Agent",
    desc: "AI-powered answers from your legal knowledge base.",
    placeholder: "Ask about contracts, liability, compliance, policy\u2026",
    es: ["\u2696\uFE0F", "Legal Agent", "Ask any legal question \u2014 AI searches 315 documents and gives a cited answer."],
  },
};

// ── API ──────────────────────────────────────────
async function api(path, body) {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const res = await fetch(path, opt);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

// ── Messages ─────────────────────────────────────
function clearMessages() {
  const m = $("#messages");
  m.innerHTML = "";
  const a = AGENTS[state.agent];
  const llmTag = state.llm && state.agent !== "hr"
    ? '<span class="badge pill ai-badge">AI powered</span>'
    : '';
  const es = el("div", "empty-state",
    `<div class="es-ico">${a.es[0]}</div><h3>${a.es[1]} ${llmTag}</h3><p>${a.es[2]}</p>`);
  m.appendChild(es);
}
function addMessage(role, content, who) {
  const m = $("#messages");
  if (m.querySelector(".empty-state")) m.innerHTML = "";
  const wrap = el("div", `msg ${role}`);
  if (who) wrap.appendChild(el("div", "who", who));
  const bubble = el("div", "bubble");
  if (typeof content === "string") bubble.innerHTML = content;
  else bubble.appendChild(content);
  wrap.appendChild(bubble);
  m.appendChild(wrap);
  m.scrollTop = m.scrollHeight;
  return wrap;
}
function addTyping() {
  const t = el("div", "msg agent");
  t.appendChild(el("div", "bubble", `<div class="typing"><span></span><span></span><span></span></div>`));
  $("#messages").appendChild(t);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return t;
}

// ── Finance rendering ────────────────────────────
function renderFinance(key, label, data) {
  const rc = el("div", "rc");
  rc.appendChild(el("div", "card-title", esc(label)));

  const stats = el("div", "stat-grid");
  const stat = (k, v, peach) => {
    const s = el("div", "stat" + (peach ? " peach" : ""));
    s.appendChild(el("div", "k", k));
    s.appendChild(el("div", "v", v));
    return s;
  };

  if (key === "summary") {
    stats.appendChild(stat("Cash balance", money(data.cash_balance)));
    stats.appendChild(stat("Total income", money(data.total_income)));
    stats.appendChild(stat("Total expense", money(data.total_expense), true));
    stats.appendChild(stat("Burn / month", money(data.monthly_burn_rate), true));
    stats.appendChild(stat("Runway", data.runway_months + " mo"));
    rc.appendChild(stats);
    if (data.top_expense_categories?.length) {
      rc.appendChild(el("div", "card-title", "Top expenses"));
      rc.appendChild(expenseBars(data.top_expense_categories));
    }
  } else if (key === "cash_balance") {
    stats.appendChild(stat("Cash balance", money(data.cash_balance)));
    rc.appendChild(stats);
  } else if (key === "burn_rate") {
    stats.appendChild(stat("Monthly burn", money(data.monthly_burn_rate), true));
    rc.appendChild(stats);
  } else if (key === "runway") {
    stats.appendChild(stat("Runway", data.runway_months + " months"));
    rc.appendChild(stats);
  } else if (key === "monthly_profit") {
    rc.appendChild(profitTable(data));
  } else if (key === "top_expenses") {
    rc.appendChild(expenseBars(data));
  }
  return rc;
}
function expenseBars(list) {
  const max = Math.max(...list.map((x) => x.total));
  const box = el("div");
  list.forEach((x) => {
    const row = el("div", "bar-row");
    row.appendChild(el("div", "lbl", esc(x.category)));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = (x.total / max) * 100 + "%";
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("div", "pct", money(x.total)));
    box.appendChild(row);
  });
  return box;
}
function profitTable(rows) {
  const wrap = el("div", "table-wrap");
  let h = `<table class="data-table"><thead><tr><th>Month</th><th>Income</th><th>Expense</th><th>Profit</th><th>Margin</th></tr></thead><tbody>`;
  rows.forEach((r) => {
    h += `<tr><td>${esc(r.month)}</td><td>${money(r.income)}</td><td>${money(r.expense)}</td><td>${money(r.profit)}</td><td>${r.margin}%</td></tr>`;
  });
  wrap.innerHTML = h + "</tbody></table>";
  return wrap;
}

// ── HR rendering ─────────────────────────────────
function renderHR(data) {
  const rc = el("div", "rc");
  const m = data.metrics;
  const vClass = m.verdict === "FAIR" ? "fair" : m.verdict === "BORDERLINE" ? "warn" : "bad";
  rc.appendChild(el("div", "card-title",
    `${esc(data.dataset_label)} \u00B7 ${data.shortlisted} shortlisted of ${data.dataset_rows}`));

  const stats = el("div", "stat-grid");
  const stat = (k, v, cls) => {
    const s = el("div", "stat" + (cls || ""));
    s.appendChild(el("div", "k", k));
    const val = el("div", "v"); val.innerHTML = v; s.appendChild(val);
    return s;
  };
  stats.appendChild(stat("Fairness verdict", `<span class="badge ${vClass}">${m.verdict}</span>`));
  stats.appendChild(stat("Disparate impact", m.di_after, " peach"));
  stats.appendChild(stat("Equal opportunity", m.equal_opportunity_di));
  stats.appendChild(stat("DI before reweight", m.di_before_reweight, " peach"));
  rc.appendChild(stats);

  rc.appendChild(el("div", "bubble", `<b>Diagnosis:</b> ${esc(m.diagnosis)}`));

  // gender breakdown bars
  rc.appendChild(el("div", "card-title", "Selection rate by group"));
  const gb = el("div");
  Object.entries(data.gender_breakdown).forEach(([g, v]) => {
    const row = el("div", "bar-row");
    row.appendChild(el("div", "lbl", `${esc(g)} (${v.selected}/${v.pool})`));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill"); fill.style.width = Math.min(v.rate, 100) + "%";
    track.appendChild(fill); row.appendChild(track);
    row.appendChild(el("div", "pct", v.rate + "%"));
    gb.appendChild(row);
  });
  rc.appendChild(gb);

  // candidates table
  rc.appendChild(el("div", "card-title", "Shortlisted candidates"));
  rc.appendChild(candidateTable(data.candidates));

  // features
  if (data.protected_removed?.length) {
    const pr = el("div");
    pr.appendChild(el("span", "card-title", "Protected/proxy columns removed: "));
    data.protected_removed.forEach((c) => pr.appendChild(el("span", "badge pill", esc(c))));
    rc.appendChild(pr);
  }
  return rc;
}
function candidateTable(cands) {
  const wrap = el("div", "table-wrap");
  if (!cands.length) { wrap.textContent = "No candidates."; return wrap; }
  const cols = Object.keys(cands[0]).filter((c) => c !== "match_score").slice(0, 5);
  let h = `<table class="data-table"><thead><tr><th>#</th>`;
  cols.forEach((c) => (h += `<th>${esc(c)}</th>`));
  h += `<th>Match</th></tr></thead><tbody>`;
  cands.slice(0, 20).forEach((c, i) => {
    h += `<tr><td>${i + 1}</td>`;
    cols.forEach((col) => (h += `<td>${esc(c[col])}</td>`));
    h += `<td><b>${c.match_score}%</b></td></tr>`;
  });
  wrap.innerHTML = h + "</tbody></table>";
  return wrap;
}

// ── Legal rendering ──────────────────────────────
function renderLegalOverview(data) {
  const rc = el("div", "rc");
  rc.appendChild(el("div", "card-title", `Legal knowledge base \u00B7 ${data.total_documents} documents`));
  const stats = el("div", "stat-grid");
  data.categories.forEach((c, i) => {
    const s = el("div", "stat" + (i % 2 ? " peach" : ""));
    s.appendChild(el("div", "k", c.label));
    s.appendChild(el("div", "v", c.count));
    stats.appendChild(s);
  });
  rc.appendChild(stats);
  const risk = el("div");
  risk.appendChild(el("div", "card-title", "Contract-review risk distribution"));
  const rd = data.risk_distribution;
  const total = Object.values(rd).reduce((a, b) => a + b, 0);
  Object.entries(rd).forEach(([k, v]) => {
    const row = el("div", "bar-row");
    row.appendChild(el("div", "lbl", esc(k) + " risk"));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill"); fill.style.width = (v / total) * 100 + "%";
    track.appendChild(fill); row.appendChild(track);
    row.appendChild(el("div", "pct", v));
    risk.appendChild(row);
  });
  rc.appendChild(risk);
  return rc;
}
function renderLegalResults(data) {
  const rc = el("div", "rc");
  if (!data.results || !data.results.length) {
    rc.appendChild(el("div", "bubble", esc(data.message || "No results.")));
    return rc;
  }
  rc.appendChild(el("div", "card-title", `${data.match_count} matches for \u201C${esc(data.query)}\u201D \u2014 top ${data.results.length}`));
  data.results.forEach((r) => {
    const item = el("div", "legal-item");
    item.appendChild(el("div", "li-head",
      `<span class="li-title">${esc(r.title)}</span><span class="badge pill">${esc(r.category)}</span>`));
    item.appendChild(el("div", "li-body", esc(r.body)));
    rc.appendChild(item);
  });
  return rc;
}

// ── LLM chat rendering ─────────────────────────
function renderMarkdown(text) {
  return esc(text)
    .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")
    .replace(/\*(.*?)\*/g, "<i>$1</i>")
    .replace(/`(.*?)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
}

function renderChatAnswer(text) {
  const rc = el("div", "rc");
  rc.appendChild(el("div", "bubble", renderMarkdown(text)));
  return rc;
}

// ── LLM chat functions ──────────────────────────
async function runFinanceChat(text) {
  addMessage("user", esc(text), state.startup);
  const t = addTyping();
  try {
    const res = await api("/api/finance/chat", { message: text });
    t.remove();
    addMessage("agent", renderChatAnswer(res.answer), "Finance Agent");
  } catch (e) {
    t.remove();
    const [key, label] = financeKeyword(text);
    runFinance(key, label);
  }
}

async function runLegalChat(text) {
  const t = addTyping();
  try {
    const res = await api("/api/legal/chat", { message: text });
    t.remove();
    const rc = el("div", "rc");
    rc.appendChild(el("div", "bubble", renderMarkdown(res.answer)));
    if (res.sources?.length) {
      rc.appendChild(el("div", "card-title", "Sources"));
      const srcWrap = el("div");
      res.sources.forEach((s) => {
        srcWrap.appendChild(el("span", "badge pill", esc(s.category || s.id || "doc")));
      });
      rc.appendChild(srcWrap);
    }
    addMessage("agent", rc, "Legal Agent");
  } catch (e) {
    t.remove();
    runLegalSearch(text);
  }
}

// ── Quick actions per agent ──────────────────────
async function loadQuickActions() {
  const qa = $("#quick-actions");
  qa.innerHTML = "";
  try {
    if (state.agent === "finance") {
      const opts = await api("/api/finance/options");
      opts.forEach((o) => {
        const chip = el("button", "chip", o.label);
        chip.onclick = () => runFinance(o.key, o.label);
        qa.appendChild(chip);
      });
    } else if (state.agent === "hr") {
      state.hrDatasets = await api("/api/hr/datasets");
      const sel = el("select", "chip select");
      state.hrDatasets.forEach((d) => sel.appendChild(el("option", null, `${d.label} (${d.file})`)));
      sel.id = "hr-dataset";
      qa.appendChild(sel);
      const count = el("select", "chip select");
      [10, 25, 50, 100].forEach((n) => count.appendChild(el("option", null, `Top ${n}`)));
      count.id = "hr-count"; count.selectedIndex = 2;
      qa.appendChild(count);
      const run = el("button", "chip peachy", "\u26A1 Run bias-free shortlist");
      run.onclick = runHR;
      qa.appendChild(run);
    } else if (state.agent === "legal") {
      const ov = await api("/api/legal/overview");
      state.legalCats = ov.categories;
      const oc = el("button", "chip", "\uD83D\uDCCA Knowledge-base overview");
      oc.onclick = () => showLegalOverview(ov);
      qa.appendChild(oc);
      ov.categories.forEach((c) => {
        const chip = el("button", "chip peachy", `${c.label} (${c.count})`);
        chip.onclick = () => runLegalCategory(c.key, c.label);
        qa.appendChild(chip);
      });
    }
  } catch (e) {
    qa.appendChild(el("div", "who", "\u26A0\uFE0F " + e.message));
  }
}

// ── Actions ──────────────────────────────────────
async function runFinance(key, label) {
  addMessage("user", esc(label), state.startup);
  const t = addTyping();
  try {
    const res = await api("/api/finance/query", { option: key });
    t.remove();
    addMessage("agent", renderFinance(key, res.label, res.data), "Finance Agent");
  } catch (e) { t.remove(); addMessage("agent", "\u26A0\uFE0F " + esc(e.message), "Finance Agent"); }
}

async function runHR() {
  const idx = $("#hr-dataset").selectedIndex;
  const ds = state.hrDatasets[idx];
  const top_n = [10, 25, 50, 100][$("#hr-count").selectedIndex];
  addMessage("user", `Shortlist top ${top_n} from ${esc(ds.label)}`, state.startup);
  const t = addTyping();
  try {
    const res = await api("/api/hr/shortlist", { dataset: ds.file, top_n });
    t.remove();
    addMessage("agent", renderHR(res), "Hiring Agent");
  } catch (e) { t.remove(); addMessage("agent", "\u26A0\uFE0F " + esc(e.message), "Hiring Agent"); }
}

async function runLegalCategory(key, label) {
  addMessage("user", `Show ${esc(label)}`, state.startup);
  const t = addTyping();
  try {
    const res = await api("/api/legal/category", { category: key });
    t.remove();
    const rc = el("div", "rc");
    rc.appendChild(el("div", "card-title", `${esc(res.category)} \u00B7 ${res.items.length} shown`));
    res.items.forEach((it) => {
      const item = el("div", "legal-item");
      item.appendChild(el("div", "li-title", esc(it.title)));
      item.appendChild(el("div", "li-body", esc(it.body)));
      rc.appendChild(item);
    });
    addMessage("agent", rc, "Legal Agent");
  } catch (e) { t.remove(); addMessage("agent", "\u26A0\uFE0F " + esc(e.message), "Legal Agent"); }
}
function showLegalOverview(ov) {
  addMessage("user", "Show knowledge-base overview", state.startup);
  addMessage("agent", renderLegalOverview(ov), "Legal Agent");
}
async function runLegalSearch(query) {
  const t = addTyping();
  try {
    const res = await api("/api/legal/query", { query });
    t.remove();
    addMessage("agent", renderLegalResults(res), "Legal Agent");
  } catch (e) { t.remove(); addMessage("agent", "\u26A0\uFE0F " + esc(e.message), "Legal Agent"); }
}

// keyword routing for typed finance messages (fallback when LLM unavailable)
function financeKeyword(text) {
  const t = text.toLowerCase();
  if (/(runway|months left)/.test(t)) return ["runway", "Runway (months left)"];
  if (/(burn)/.test(t)) return ["burn_rate", "Monthly burn rate"];
  if (/(cash|balance)/.test(t)) return ["cash_balance", "Current cash balance"];
  if (/(profit|margin)/.test(t)) return ["monthly_profit", "Monthly profit & margin"];
  if (/(expense|spend|cost)/.test(t)) return ["top_expenses", "Top expense categories"];
  return ["summary", "Full financial overview"];
}

// ── Composer ─────────────────────────────────────
function onSend(e) {
  e.preventDefault();
  const input = $("#composer-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";

  if (state.agent === "finance") {
    if (state.llm) {
      runFinanceChat(text);
    } else {
      const [key, label] = financeKeyword(text);
      runFinance(key, label);
    }
  } else if (state.agent === "legal") {
    addMessage("user", esc(text), state.startup);
    if (state.llm) {
      runLegalChat(text);
    } else {
      runLegalSearch(text);
    }
  } else if (state.agent === "hr") {
    addMessage("user", esc(text), state.startup);
    if (/shortlist|candidate|hir|rank/i.test(text) && state.hrDatasets.length) {
      const ds = state.hrDatasets[/attrition|ibm/i.test(text) ? Math.min(1, state.hrDatasets.length - 1) : 0];
      const t = addTyping();
      api("/api/hr/shortlist", { dataset: ds.file, top_n: 50 })
        .then((res) => { t.remove(); addMessage("agent", renderHR(res), "Hiring Agent"); })
        .catch((err) => { t.remove(); addMessage("agent", "\u26A0\uFE0F " + esc(err.message), "Hiring Agent"); });
    } else {
      addMessage("agent", "Pick a dataset and hit <b>Run bias-free shortlist</b> below, or say \u201Cshortlist candidates from the income dataset\u201D.", "Hiring Agent");
    }
  }
}

// ── Agent switching ──────────────────────────────
function switchAgent(agent) {
  state.agent = agent;
  document.querySelectorAll(".agent-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.agent === agent));
  const a = AGENTS[agent];
  $("#agent-title").textContent = a.title;
  $("#agent-desc").textContent = a.desc;
  $("#composer-input").placeholder = a.placeholder;
  clearMessages();
  loadQuickActions();
}

// ── Init ─────────────────────────────────────────
async function initApp() {
  try {
    const llmStatus = await api("/api/llm/status");
    state.llm = llmStatus.available;
  } catch (e) {
    state.llm = false;
  }
  const llmDot = $(".status-dot");
  if (llmDot && state.llm) {
    llmDot.classList.add("ai-active");
    $(".sidebar-foot").innerHTML = '<div class="status-dot ai-active"></div> Agents online \u00B7 AI chat active';
  }
  document.querySelectorAll(".agent-btn").forEach((b) =>
    (b.onclick = () => switchAgent(b.dataset.agent)));
  $("#composer-form").addEventListener("submit", onSend);
  $("#new-chat").onclick = () => clearMessages();
  switchAgent("finance");
}

$("#login-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const name = $("#startup-name").value.trim();
  if (!name) return;
  state.startup = name;
  $("#brand-name").textContent = name;
  const ov = $("#login");
  ov.classList.add("fade");
  setTimeout(() => {
    ov.classList.add("hidden");
    $("#app").classList.remove("hidden");
    initApp();
  }, 550);
});
