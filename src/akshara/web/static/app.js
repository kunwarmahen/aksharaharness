/* akshara web UI — a dumb renderer over the envelope protocol.
   One websocket carries everything; REST handles plain controls.
   On every (re)connect the transcript is rebuilt from the server's
   history replay, so reloads and reconnects are always consistent. */

"use strict";

const $ = (sel) => document.querySelector(sel);
const transcript = $("#transcript");

const ui = {
  ws: null,
  connected: false,
  turnActive: false,
  staged: [],            // [{filename, data_base64, url}]
  currentAssistant: null, // streaming text node
  assistantText: "",      // raw markdown accumulated for it
  renderQueued: false,    // rAF throttle for live re-rendering
  currentThinking: null,
  openToolCard: null,     // tool_start awaiting its result
  modalId: null,
  lastState: null,        // most recent state envelope (for mid-turn merges)
};

function forgetAssistant() {
  ui.currentAssistant = null;
  ui.assistantText = "";
}

/* ---------- connection ---------- */

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  ui.ws = ws;

  ws.onopen = () => { setOffline(false); };
  ws.onclose = () => {
    setOffline(true);
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();

  ws.onmessage = (m) => {
    let env;
    try { env = JSON.parse(m.data); } catch { return; }
    route(env);
  };
}

function send(obj) {
  if (ui.ws && ui.ws.readyState === WebSocket.OPEN) {
    ui.ws.send(JSON.stringify(obj));
  }
}

function setOffline(off) {
  $("#offline").classList.toggle("hidden", !off);
  if (off) { ui.connected = false; }
}

/* ---------- envelope routing ---------- */

function route(env) {
  switch (env.type) {
    case "state":          onState(env); break;
    case "turn_started":   beginTurn(); break;
    case "start":          setStatus(`· ${env.model}`); break;

    case "delta":              appendDelta(env.text); break;
    case "thinking_delta":     appendThinking(env.text); break;
    case "redacted_thinking":  addRedacted(env.chars); break;

    case "tool_start":         openToolCard(env.name); break;
    case "tool_result":        fillToolCard(env);
                               // live envelopes carry fresh pressure; replay
                               // ones don't -- only merge when present
                               if (env.utilization != null && ui.lastState) {
                                 renderPressure(
                                   {...ui.lastState, utilization: env.utilization});
                               }
                               break;

    case "permission_request": showPermissionModal(env); break;
    case "ask":                showAskModal(env); break;
    case "resolved":           closeModalIf(env.id); break;

    case "turn_end":       onTurnEnd(env); break;
    case "turn_error":     addBanner(env.message, true); break;
    case "turn_cancelled": addBanner("turn cancelled", false, true); break;
    case "turn_done":      endTurn(); break;

    /* history replay shapes */
    case "user_message":    addUserMessage(env.text, env.images, true); break;
    case "assistant_text":  assistantTextDone(env.text); break;
    case "thinking_done":   addThinkingDone(env.text); break;
  }
}

function onState(env) {
  // First envelope of a connection: rebuild the transcript from scratch so
  // reconnects never duplicate what we already rendered live.
  if (!ui.connected) {
    ui.connected = true;
    transcript.textContent = "";
    forgetAssistant();
    ui.currentThinking = ui.openToolCard = null;
    fetchHistory();
  }
  applyHeader(env);
}

async function fetchHistory() {
  try {
    const res = await fetch("/api/history");
    if (!res.ok) return;
    for (const env of await res.json()) route(env);
    scrollDown(true);
  } catch { /* offline; retry happens via reconnect */ }
}

function applyHeader(s) {
  ui.lastState = s;
  $("#provider-name").textContent = s.provider;
  $("#model-name").textContent = s.model;
  const yolo = s.mode === "yolo";
  $("#mode-label").textContent = s.mode || "ask";
  $("#chip-mode").classList.toggle("chip-yolo", yolo);
  // session awareness chip: label is the level, tooltip carries the exact
  // fact sheet the model sees (null when the host built no EnvContext)
  const ec = s.env_context;
  $("#env-label").textContent = ec ? ec.mode : "—";
  $("#chip-env").classList.toggle("chip-env-full", !!ec && ec.mode === "full");
  $("#chip-env").title = (ec && ec.block)
    ? `session awareness — click to cycle off ⇄ local ⇄ full\n\n${ec.block}`
    : "session awareness — click cycles off ⇄ local ⇄ full";
  $("#chip-cost").textContent = s.cost_line || "";
  // tools chip: "N" live, "+M off" when the operator pulled some
  const off = (s.disabled_tools || []).length;
  $("#tools-count").textContent =
    (s.tools ? s.tools.length : "—") + (off ? ` · ${off} off` : "");
  renderPressure(s);
  setTurnUI(s.turn_active);
}

/* Context-window pressure. The bar mirrors auto-compaction's thresholds
   (60% yellow / 80% red); under 10% we show a decimal so early-session
   readings aren't all rounded to a lying flat 0%. The tooltip carries
   the honest numbers behind it (real provider count once one exists,
   chars/4 estimate before that). */
function renderPressure(s) {
  const util = s.utilization;
  $("#chip-ctxbar").classList.toggle("hidden", util == null);
  if (util == null) return;
  const pct = util * 100;
  const shown = pct < 10 ? pct.toFixed(1) : String(Math.round(pct));
  $("#ctx-fill").style.width = `${Math.min(pct, 100)}%`;
  $("#ctx-fill").className = "ctxbar-fill"
    + (pct >= 80 ? " danger" : pct >= 60 ? " warn" : "");
  $("#ctx-pct").textContent = `${shown}%`;
  $("#chip-ctxbar").classList.toggle("chip-ctx-danger", pct >= 80);
  $("#chip-ctxbar").title =
    `context pressure — ~${fmtNum(s.context_tokens)} of ` +
    `${fmtNum(s.context_window)} window tokens` +
    (s.context_estimated ? " (estimated)" : "") +
    "; auto-compact at 80%";
}

function fmtNum(n) {
  return n >= 10_000 ? `${(n / 1000).toFixed(n % 1000 ? 1 : 0)}k`
    : String(n ?? 0);
}

/* ---------- turn lifecycle ---------- */

function beginTurn() {
  ui.turnActive = true;
  forgetAssistant();
  ui.currentThinking = ui.openToolCard = null;
  setStatus("connecting…");
  setTurnUI(true);
}

function endTurn() {
  ui.turnActive = false;
  forgetAssistant();
  ui.currentThinking = ui.openToolCard = null;
  hideStatus();
  setTurnUI(false);
}

function setTurnUI(active) {
  ui.turnActive = active;
  $("#btn-send").disabled = active;
  $("#btn-cancel").classList.toggle("hidden", !active);
  if (!active) hideStatus();
}

function setStatus(text) {
  $("#statusline").classList.remove("hidden");
  $("#status-text").textContent = text;
}
function hideStatus() { $("#statusline").classList.add("hidden"); }

function scrollDown(force) {
  const nearBottom =
    transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 160;
  if (force || nearBottom) transcript.scrollTop = transcript.scrollHeight;
}

/* ---------- message rendering ---------- */

function addUserMessage(text, images = 0, replay = false) {
  const wrap = document.createElement("div");
  wrap.className = "msg-user";
  const label = document.createElement("div");
  label.className = "label";
  label.textContent = "you";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.append(label, bubble);
  if (images) {
    const note = document.createElement("div");
    note.className = "imgs";
    note.textContent = `🖼 ${images} image${images > 1 ? "s" : ""} attached`;
    wrap.append(note);
  }
  transcript.append(wrap);
  if (!replay) scrollDown();
  return bubble;
}

function ensureAssistant() {
  if (!ui.currentAssistant) {
    const wrap = document.createElement("div");
    wrap.className = "msg-assistant";
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = "agent";
    const textEl = document.createElement("div");
    textEl.className = "text empty";
    wrap.append(label, textEl);
    transcript.append(wrap);
    ui.currentAssistant = textEl;
  }
  return ui.currentAssistant;
}

function appendDelta(text) {
  ui.assistantText += text;
  const el = ensureAssistant();
  el.classList.remove("empty");
  // Live markdown re-render, throttled to one per animation frame:
  // a partial document (half a table, an open fence) renders as far as
  // it goes, and turn_end's full-text pass settles any artifact.
  if (!ui.renderQueued) {
    ui.renderQueued = true;
    requestAnimationFrame(() => {
      ui.renderQueued = false;
      if (ui.currentAssistant) {
        ui.currentAssistant.innerHTML = renderMarkdown(ui.assistantText);
        scrollDown();
      }
    });
  }
}

function assistantTextDone(text) {
  const el = ensureAssistant();
  el.classList.remove("empty");
  ui.assistantText = text; // replay replaces rather than appends
  el.innerHTML = renderMarkdown(text);
  scrollDown();
}

function ensureThinking() {
  if (!ui.currentThinking) {
    const el = document.createElement("div");
    el.className = "thinking";
    transcript.append(el);
    ui.currentThinking = el;
  }
  return ui.currentThinking;
}
function appendThinking(text) { ensureThinking().textContent += text; scrollDown(); }
function addThinkingDone(text) { ensureThinking().textContent = text; }

function addRedacted(chars) {
  const el = document.createElement("div");
  el.className = "redacted";
  el.textContent = `· redacted reasoning (${chars} chars, encrypted)`;
  transcript.append(el);
  scrollDown();
}

/* ---------- tool cards ---------- */

const OUTPUT_PREVIEW = 400;

function openToolCard(name) {
  ui.currentThinking = null; // thinking block ends where tools begin
  forgetAssistant();
  const card = document.createElement("div");
  card.className = "tool-card running";
  card.innerHTML = `
    <div class="tool-head"><span class="tool-head">${esc(name)}()</span>
    <span class="tool-status">running…</span></div>`;
  transcript.append(card);
  ui.openToolCard = card;
  scrollDown();
}

function fillToolCard(env) {
  let card = ui.openToolCard;
  // Replay has no tool_start before results — create as needed.
  if (!card) {
    openToolCard(env.name);
    card = ui.openToolCard;
  }
  card.classList.remove("running");
  card.classList.toggle("error", !!env.is_error);
  card.querySelector(".tool-status").textContent =
    env.is_error ? "error" : "ok";
  card.querySelector(".tool-head span").textContent = `${env.name}()`;

  const body = document.createElement("div");
  body.className = "tool-body";

  const argsDetails = document.createElement("details");
  argsDetails.innerHTML = "<summary>arguments</summary>";
  const argsPre = document.createElement("pre");
  argsPre.className = "args";
  argsPre.textContent = JSON.stringify(env.arguments ?? {}, null, 2);
  argsDetails.append(argsPre);
  body.append(argsDetails);

  const outPre = document.createElement("pre");
  outPre.className = "output";
  const output = String(env.output ?? "");
  if (output.length > OUTPUT_PREVIEW && !env.is_error) {
    outPre.textContent = output.slice(0, OUTPUT_PREVIEW);
    const more = document.createElement("span");
    more.className = "output-clipped";
    more.textContent =
      `\n[… ${output.length - OUTPUT_PREVIEW} more chars — click to expand]`;
    more.onclick = () => { outPre.textContent = output; more.remove(); };
    body.append(outPre, more);
  } else {
    outPre.textContent = output;
    body.append(outPre);
  }
  card.append(body);
  ui.openToolCard = null;
  scrollDown();
}

/* ---------- banners / footer ---------- */

function addBanner(message, isError, cancelledStyle = false) {
  const el = document.createElement("div");
  el.className = isError ? "banner banner-error"
    : cancelledStyle ? "banner-cancelled" : "banner";
  el.textContent = message;
  transcript.append(el);
  scrollDown();
}

function onTurnEnd(env) {
  if (env.reason !== "end_turn") {
    addBanner(`── turn ended: ${env.reason} (after ${env.iterations} iteration(s))`,
              false);
    return;
  }
  if (env.text && env.text.trim()) {
    assistantTextDone(env.text); // final full-text pass: settles streaming
  } else {
    // end_turn with only thinking/tool noise -- say so rather than
    // leaving the operator staring at silence (mirrors render.py)
    const el = ensureAssistant();
    el.classList.remove("empty");
    if (!ui.assistantText.trim()) el.textContent = "(no text in reply)";
  }
  const foot = document.createElement("div");
  foot.className = "turn-foot";
  const cost = env.cost_line ? ` · ~cost: ${env.cost_line}` : "";
  foot.textContent =
    `── ${env.stop_reason} · ${env.input_tokens} in / ${env.output_tokens} out` +
    `${cost} · ${env.iterations} iteration(s)`;
  const wrap = ensureAssistant().parentElement;
  wrap.append(foot);
  ui.currentAssistant = null;
  scrollDown();
}

/* ---------- modals ---------- */

function showModal(html) {
  $("#modal").innerHTML = html;
  $("#modal-backdrop").classList.remove("hidden");
}

function closeModalIf(id) {
  if (ui.modalId === id) {
    ui.modalId = null;
    $("#modal-backdrop").classList.add("hidden");
  }
}

function showPermissionModal(env) {
  ui.modalId = env.id;
  showModal(`
    <div class="kind-tag">approval needed</div>
    <h3>${esc(env.tool_name)}() ${env.edited
      ? '<span class="edited-flag">(edited)</span>' : ""}</h3>
    <div class="summary">${esc(env.summary)}</div>
    <details><summary>raw arguments</summary>
      <pre class="args">${esc(JSON.stringify(env.arguments ?? {}, null, 2))}</pre>
    </details>
    <div class="modal-actions">
      <button class="m-btn" data-act="edit">edit</button>
      <button class="m-btn danger" data-act="deny">deny</button>
      <button class="m-btn primary" data-act="approve">approve ⏎</button>
    </div>`);
  const answer = (payload) => send({ type: "answer", id: env.id, ...payload });
  const onKey = (e) => {
    // Enter approves only while THIS modal is up and no edit box is open
    if (ui.modalId !== env.id || e.key !== "Enter") return;
    if ($("#modal textarea")) return;
    e.preventDefault();
    answer({ decision: "approve" });
  };
  document.addEventListener("keydown", onKey);
  $("#modal").onclick = (e) => {
    const act = e.target?.dataset?.act;
    if (!act) return;
    if (act === "approve") { document.removeEventListener("keydown", onKey); answer({ decision: "approve" }); }
    if (act === "deny") { document.removeEventListener("keydown", onKey); answer({ decision: "deny" }); }
    if (act === "edit") renderEditBox(env, answer, onKey);
  };
}

function renderEditBox(env, answer, dismissKeyHandler) {
  const box = document.createElement("div");
  box.innerHTML = `
    <textarea id="edit-args">${esc(JSON.stringify(env.arguments ?? {}, null, 2))}</textarea>
    <div class="modal-actions">
      <button class="m-btn" id="edit-cancel">back</button>
      <button class="m-btn primary" id="edit-review">review edited call</button>
    </div>`;
  [...$("#modal").children].forEach((c) => {
    if (!c.classList.contains("kind-tag")) c.remove();
  });
  $("#modal").append(box);
  box.querySelector("#edit-cancel").onclick = () => showPermissionModal(env);
  box.querySelector("#edit-review").onclick = () => {
    try {
      const edited = JSON.parse(box.querySelector("#edit-args").value);
      if (edited === null || typeof edited !== "object"
          || Array.isArray(edited)) throw new Error("must be a JSON object");
      if (dismissKeyHandler) document.removeEventListener("keydown", dismissKeyHandler);
      answer({ decision: "edit", edited_args: edited });
      // server re-sends an updated permission_request; nothing else to do
    } catch (err) {
      alert(`bad edit: ${err.message}`);
    }
  };
}

function showAskModal(env) {
  ui.modalId = env.id;
  const choices = env.choices || [];
  const rows = choices.map((choice, i) => `
    <label class="choice-row">
      <input type="radio" name="ask-choice" value="${i}" ${i === 0 ? "checked" : ""}>
      ${esc(choice)}</label>`).join("");
  showModal(`
    <div class="kind-tag">the agent asks</div>
    <h3>${esc(env.question)}</h3>
    ${env.context ? `<div class="context-note">${esc(env.context)}</div>` : ""}
    ${rows ? `<div id="choices">${rows}</div>` : ""}
    <textarea id="ask-text" placeholder="${choices
      ? "or type your own answer…" : "your answer…"}"></textarea>
    <div class="modal-actions">
      <button class="m-btn primary" id="ask-send">send answer</button>
    </div>`);
  const commit = () => {
    let text;
    const picked = $("#modal input[name=ask-choice]:checked");
    const typed = $("#ask-text").value.trim();
    if (typed) text = typed;
    else if (picked) text = choices[Number(picked.value)];
    else return;
    send({ type: "answer", id: env.id, text });
  };
  $("#ask-send").onclick = commit;
  $("#ask-text").focus();
  $("#ask-text").onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commit(); }
  };
}

/* ---------- composer ---------- */

const input = $("#input");

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
input.addEventListener("input", autoGrow);
function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}

$("#btn-send").onclick = sendMessage;
$("#btn-cancel").onclick = () => send({ type: "cancel" });
$("#btn-image").onclick = () => $("#file-input").click();

$("#file-input").onchange = async () => {
  for (const file of $("#file-input").files) {
    if (ui.staged.length >= 5) { toast("5 images max per message"); break; }
    try {
      const buf = await file.arrayBuffer();
      if (buf.byteLength > 5 * 1024 * 1024) { toast(`${file.name}: over 5 MB`); continue; }
      ui.staged.push({
        filename: file.name,
        data_base64: b64FromBuffer(buf),
        url: URL.createObjectURL(file),
      });
    } catch { toast(`could not read ${file.name}`); }
  }
  $("#file-input").value = "";
  renderStaged();
};

function b64FromBuffer(buf) {
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

function renderStaged() {
  const wrap = $("#staged");
  wrap.classList.toggle("hidden", !ui.staged.length);
  wrap.textContent = "";
  ui.staged.forEach((img, i) => {
    const chip = document.createElement("span");
    chip.className = "staged-chip";
    const thumb = document.createElement("img");
    thumb.src = img.url;
    chip.append(thumb, document.createTextNode(img.filename));
    const x = document.createElement("button");
    x.textContent = "×";
    x.title = "remove";
    x.onclick = () => { ui.staged.splice(i, 1); renderStaged(); };
    chip.append(x);
    wrap.append(chip);
  });
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text || ui.turnActive) return;
  const body = { text, images: ui.staged.map(({ filename, data_base64 }) =>
    ({ filename, data_base64 })) };
  try {
    const res = await fetch("/api/message", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.status === 409) { toast("a turn is already running"); return; }
    if (!res.ok) {
      toast((await res.json().catch(() => ({}))).detail || "failed to send");
      return;
    }
  } catch { toast("offline — not sent"); return; }
  addUserMessage(text, ui.staged.length);
  ui.staged.forEach((img) => URL.revokeObjectURL(img.url));
  ui.staged = [];
  renderStaged();
  input.value = "";
  autoGrow();
  beginTurn();
  setStatus("waiting for first token…");
}

/* ---------- header controls ---------- */

async function post(path, payload = {}) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) { toast(data.detail || res.statusText); return null; }
  return data;
}

$("#chip-model").onclick = () => {
  const slug = prompt("model slug:", $("#model-name").textContent);
  if (slug) post("/api/model", { model: slug }).then((s) => s && applyHeader(s));
};
// esc stops the run — same as the red button. Deliberately NOT bound while
// a permission/ask modal is up: there, esc must not silently deny anything.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && ui.turnActive && ui.modalId === null) {
    e.preventDefault();
    send({ type: "cancel" });
  }
});
// Permission mode: flip ask <-> yolo. Allowed mid-turn on purpose -- the
// server applies it to every not-yet-approved call of the running turn.
$("#chip-mode").onclick = () => {
  const next = $("#mode-label").textContent === "yolo" ? "ask" : "yolo";
  post("/api/permissions", { mode: next }).then((s) => {
    if (!s) return;
    applyHeader(s);
    toast(next === "yolo"
      ? "yolo on — tools run without asking"
      : "permission prompts back on");
  });
};
// Session awareness: cycle off -> local -> full -> off. Mid-turn allowed on
// purpose -- the recomposed system lands on the turn's next model call.
const ENV_CYCLE = { off: "local", local: "full", full: "off" };
$("#chip-env").onclick = () => {
  const next = ENV_CYCLE[$("#env-label").textContent] || "local";
  post("/api/env-context", { mode: next }).then((s) => {
    if (!s) return;
    applyHeader(s);
    toast(`env context: ${next}`);
  });
};
$("#chip-provider").onclick = () => {
  const name = prompt(
    "provider (anthropic | openai | responses | ollama):",
    $("#provider-name").textContent);
  if (name) post("/api/provider", { provider: name }).then((s) => s && applyHeader(s));
};
$("#btn-save").onclick = async () => {
  const d = await post("/api/save", { name: prompt("checkpoint name:", "default") || "default" });
  if (d) toast(`saved '${d.saved}' v${d.version}`);
};
$("#btn-load").onclick = async () => {
  const d = await post("/api/load", { name: prompt("checkpoint name:", "default") || "default" });
  if (d) {
    applyHeader(d);
    transcript.textContent = "";
    forgetAssistant();
  ui.currentThinking = ui.openToolCard = null;
    await fetchHistory();
    toast("restored");
  }
};
$("#btn-compact").onclick = async () => {
  const d = await post("/api/compact");
  if (d) {
    applyHeader(d);
    const st = d.stats;
    toast(`compacted ${st.messages_before} → ${st.messages_after} message(s)`
          + (st.masked ? `, elided ${st.masked} tool result(s)` : ""));
  }
};
$("#btn-clear").onclick = async () => {
  if (!confirm("clear conversation history?")) return;
  const d = await post("/api/clear");
  if (d) {
    applyHeader(d);
    transcript.textContent = "";
    forgetAssistant();
  ui.currentThinking = ui.openToolCard = null;
    toast("history cleared");
  }
};

/* ---------- tools panel ---------- */

/* Two sections: mcp servers on top (connect new ones, soft-toggle or
   remove existing), then every registered tool with a live switch each.
   Toggles POST immediately and are safe mid-turn: the loop consults the
   registry per call, so pulling `bash` cuts the very next bash call of a
   running turn. The panel sits UNDER the approval modal on purpose -- an
   approval can still pop over it while you're in here. */

$("#chip-tools").onclick = openToolsPanel;
$("#tools-close").onclick = closeToolsPanel;
$("#tools-backdrop").addEventListener("click", (e) => {
  if (e.target === $("#tools-backdrop")) closeToolsPanel();
});

async function openToolsPanel() {
  $("#tools-backdrop").classList.remove("hidden");
  const rows = $("#tools-rows");
  rows.textContent = "loading…";

  let servers = null, tools = null;
  try {
    const [mcpRes, toolRes] = await Promise.all([
      fetch("/api/mcp"), fetch("/api/tools")]);
    // A session without an mcp manager answers 400 -- hide the section.
    servers = mcpRes.ok ? (await mcpRes.json()).servers : null;
    tools = toolRes.ok ? await toolRes.json() : null;
  } catch { /* both stay null -> per-section error text */ }

  $("#mcp-section").classList.toggle("hidden", servers === null);
  collapseAddForm();
  renderMCPRows(servers);

  if (!tools) { rows.textContent = "could not load tools"; return; }
  rows.textContent = "";
  for (const t of tools) rows.append(toolRow(t));
}

/* ---- mcp servers ---- */

async function renderMCPRows(servers) {
  const rows = $("#mcp-rows");
  if (!servers) { rows.textContent = ""; return; } // no manager / fetch failed
  rows.textContent = "";
  if (!servers.length) {
    rows.innerHTML = '<div class="mcp-empty">no servers connected — add one above</div>';
    return;
  }
  for (const s of servers) rows.append(mcpRow(s));
}

function mcpRow(s) {
  const row = document.createElement("div");
  row.className = "mcp-row" + (s.enabled === false ? " off" : "");

  const dot = document.createElement("span");
  dot.className = "dot" + (s.healthy ? "" : " dead");
  dot.title = s.healthy ? "connected" : "not responding";
  row.append(dot);

  const name = document.createElement("div");
  name.innerHTML = `<span class="t-name">${esc(s.name)}</span>`
    + `<span class="t-badge">${esc(s.transport)}</span>`
    + (s.remembered ? '<span class="t-ro">saved</span>' : "");
  name.title = s.target;
  row.append(name);

  const count = document.createElement("div");
  count.className = "t-desc";
  count.textContent = `${s.tools} tool(s)` +
    (s.disabled ? `, ${s.disabled} off` : "") + (s.healthy ? "" : " — unreachable");
  row.append(count);

  const sw = document.createElement("label");
  sw.className = "t-switch";
  const box = document.createElement("input");
  box.type = "checkbox";
  // on while ANY of the server's tools is active; flipping off pulls all
  box.checked = s.disabled < s.tools;
  box.title = "soft-switch this server's whole toolset";
  box.onchange = async () => {
    const ok = await post("/api/mcp/toggle", { name: s.name, enabled: box.checked });
    if (!ok) { box.checked = !box.checked; return; } // refused; revert
    applyHeader(ok);
    refreshMCPRows();
    toast(box.checked ? `${s.name} back on` :
      `${s.name} off — its tools now fail as data until re-enabled`);
  };
  const knob = document.createElement("span");
  knob.className = "knob";
  sw.append(box, knob);
  row.append(sw);

  const rm = document.createElement("button");
  rm.className = "m-btn danger mcp-remove";
  rm.textContent = "✕";
  rm.title = "disconnect this server and remove its tools";
  rm.onclick = async () => {
    if (rm.dataset.armed !== "1") {          // two-step confirm: no native dialogs
      rm.dataset.armed = "1";
      rm.textContent = "sure?";
      setTimeout(() => { rm.dataset.armed = ""; rm.textContent = "✕"; }, 2500);
      return;
    }
    const out = await post("/api/mcp/remove", { name: s.name });
    if (!out) return;
    applyHeader(out);
    toast(`removed ${s.name} — ${out.removed} tool(s) gone`);
    refreshMCPRows();
  };
  row.append(rm);
  return row;
}

async function refreshMCPRows() {
  try {
    const res = await fetch("/api/mcp");
    if (res.ok) await renderMCPRows((await res.json()).servers);
  } catch { /* panel may be closing anyway */ }
}

/* the add form: fields mode by default, paste-json for power users */

let mcpMode = "fields";
$("#mcp-add-btn").onclick = () => {
  $("#mcp-add-form").classList.remove("hidden");
  $("#mcp-add-btn").classList.add("hidden");
  $("#mcp-error").classList.add("hidden");
};
$("#mcp-cancel-add").onclick = collapseAddForm;
document.querySelectorAll(".m-tab").forEach((tab) => {
  tab.onclick = () => {
    mcpMode = tab.dataset.mode;
    document.querySelectorAll(".m-tab").forEach((t) =>
      t.classList.toggle("active", t === tab));
    $("#mcp-fields").classList.toggle("hidden", mcpMode !== "fields");
    $("#mcp-json").classList.toggle("hidden", mcpMode !== "json");
  };
});
document.querySelectorAll('input[name="mcp-transport"]').forEach((radio) => {
  radio.onchange = () => {
    const http = document.querySelector('input[name="mcp-transport"]:checked').value === "http";
    $("#mcp-command").classList.toggle("hidden", http);
    $("#mcp-url").classList.toggle("hidden", !http);
  };
});

function collapseAddForm() {
  $("#mcp-add-form").classList.add("hidden");
  $("#mcp-add-btn").classList.remove("hidden");
}

// whitespace split that respects "quoted arguments" -- good enough for
// commands with spaces in paths; exotic shells should use the json mode
function splitCommand(text) {
  const out = []; let cur = "", q = null;
  for (const ch of text.trim()) {
    if (q) { if (ch === q) q = null; else cur += ch; }
    else if (ch === '"' || ch === "'") q = ch;
    else if (/\s/.test(ch)) { if (cur) { out.push(cur); cur = ""; } }
    else cur += ch;
  }
  if (cur) out.push(cur);
  return out;
}

$("#mcp-submit").onclick = async () => {
  const err = $("#mcp-error");
  err.classList.add("hidden");
  const remember = $("#mcp-remember").checked;
  const body = mcpMode === "json"
    ? { config: $("#mcp-json-text").value, remember }
    : (() => {
        const transport = document.querySelector(
          'input[name="mcp-transport"]:checked').value;
        const b = { name: $("#mcp-name").value, remember };
        if (transport === "http") b.url = $("#mcp-url").value.trim();
        else {
          const parts = splitCommand($("#mcp-command").value);
          if (!parts.length) {
            err.textContent = "enter the command to run";
            err.classList.remove("hidden");
            return null;
          }
          b.command = parts[0];
          if (parts.length > 1) b.args = parts.slice(1);
        }
        const env = {};
        for (const line of $("#mcp-env").value.split("\n")) {
          const i = line.indexOf("=");
          if (i > 0 && line.slice(0, i).trim()) env[line.slice(0, i).trim()] = line.slice(i + 1);
        }
        if (Object.keys(env).length) b.env = env;
        return b;
      })();
  if (body === null) return;

  const out = await post("/api/mcp/add", body);
  if (!out) return; // validation failure -- post() already toasted the detail
  const failed = (out.results || []).filter((r) => !r.ok);
  const added = (out.results || []).filter((r) => r.ok);
  applyHeader(out);
  if (added.length) {
    toast(added.map((r) => `${r.name}: ${r.tools} tool(s)`).join(", ")
      + ` connected${remember ? " · saved" : ""}`);
    collapseAddForm();
    $("#mcp-name").value = $("#mcp-command").value =
      $("#mcp-url").value = $("#mcp-env").value = $("#mcp-json-text").value = "";
  }
  if (failed.length) {
    err.textContent = failed.map((r) => `${r.name}: ${r.error}`).join("\n");
    err.classList.remove("hidden");
  }
  refreshMCPRows();
};

function toolRow(t) {
  const row = document.createElement("div");
  row.className = "tool-row" + (t.enabled ? "" : " off");

  const name = document.createElement("div");
  name.innerHTML = `<span class="t-name">${esc(t.name)}</span>`
    + (t.read_only ? '<span class="t-ro">read-only</span>' : "");
  row.append(name);

  const desc = document.createElement("div");
  desc.className = "t-desc";
  desc.textContent = t.description;
  row.append(desc);

  const sw = document.createElement("label");
  sw.className = "t-switch";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = t.enabled;
  box.title = t.enabled ? "disable this tool" : "re-enable this tool";
  box.onchange = async () => {
    const s = await post("/api/tools", { name: t.name, enabled: box.checked });
    if (!s) { box.checked = !box.checked; return; } // server refused; revert
    applyHeader(s);
    row.classList.toggle("off", !box.checked);
    toast(`${t.name} ${box.checked ? "enabled" : "disabled — its calls now "
      + "fail as data until re-enabled"}`);
  };
  const knob = document.createElement("span");
  knob.className = "knob";
  sw.append(box, knob);
  row.append(sw);
  return row;
}

function closeToolsPanel() {
  $("#tools-backdrop").classList.add("hidden");
}

let toastTimer = null;
function toast(msg) {
  let el = $("#toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    el.style.cssText =
      "position:fixed;bottom:74px;right:18px;background:var(--bg-raised);" +
      "border:1px solid var(--line);border-radius:8px;padding:8px 14px;" +
      "font-size:13.5px;box-shadow:0 4px 14px rgba(0,0,0,.12);z-index:99;";
    document.body.append(el);
  }
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.remove(), 3200);
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

connect();
