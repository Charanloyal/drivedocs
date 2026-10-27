/* DriveDocs AI web client. All dynamic content is rendered with DOM APIs and textContent (never innerHTML). */
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const CITATION_RE = /\[Source:\s*([^,\]\n]+?)\s*,\s*Page\s*(\d+|N\/A)\s*\]|\[unverified citation removed\]/g;
const HISTORY_KEY = "drivedocs.history.v1";
const EXAMPLES = [
  { tag: "Manual · page 3 + placard", q: "What is the recommended cold tire pressure for the rear tires?" },
  { tag: "Manual · page 5 + fuse diagram", q: "Which fuse protects the rear window defroster?" },
  { tag: "Diagram search", q: "Show the fuse box layout diagram" },
  { tag: "Troubleshooting guide", q: "The car will not power on and the touchscreen is dark. What should I do?" },
  { tag: "PII redaction", q: "What is the customer's email and phone number on the service record?" },
  { tag: "Not in the manual", q: "When should the spark plugs be replaced?" },
];
const REASONS = {
  strong_clip_match: "Strong visual similarity (CLIP ≥ image threshold)",
  visual_query_match: "You asked for a visual and the similarity passed the visual-query threshold",
  same_page_as_text_evidence: "On the same page as a cited passage",
  caption_match: "Its caption matches your request",
  image_similarity: "Visual similarity to your image",
};

const state = { info: null, ready: null, docs: [], last: null, busy: false };

/* ---------------------------------------------------------------- helpers */

function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "on") for (const [evt, fn] of Object.entries(value)) el.addEventListener(evt, fn);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key === "style") Object.assign(el.style, value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

class ApiError extends Error {
  constructor(status, code, message, body) { super(message); this.status = status; this.code = code; this.body = body; }
}

async function request(method, path, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
  let res;
  try { res = await fetch(path, opts); }
  catch (err) { throw new ApiError(0, "network_error", "Cannot reach the server. Check your connection and retry."); }
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) {
    const d = data && data.detail;
    const message = (d && (d.message || (typeof d === "string" ? d : null))) || `Request failed (HTTP ${res.status}).`;
    throw new ApiError(res.status, (d && d.code) || "http_error", message, data);
  }
  return data;
}
const api = {
  get: (p) => request("GET", p),
  post: (p, b) => request("POST", p, b),
  del: (p) => request("DELETE", p),
};

function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.className = "toast" + (isError ? " err" : "");
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, 4200);
}

function fmtScore(v) { return (typeof v === "number") ? v.toFixed(2) : "–"; }
function fmtPct(v) { return (typeof v === "number") ? `${(v * 100).toFixed(1)}%` : "n/a"; }
function fmtBytes(n) {
  if (!n && n !== 0) return "–";
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}
function fmtDate(iso) {
  const d = new Date(iso);
  return isNaN(d) ? "–" : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
function typeLabel(t) { return { pdf: "PDF", md: "Markdown", txt: "Text", png: "PNG", jpg: "JPEG", jpeg: "JPEG" }[t] || (t || "?").toUpperCase(); }

function scoreBar(value, threshold) {
  const pct = Math.max(0, Math.min(1, value || 0)) * 100;
  return h("span", { class: "score", title: threshold ? `similarity ${fmtScore(value)} (threshold ${fmtScore(threshold)})` : `similarity ${fmtScore(value)}` },
    h("span", { class: "bar" }, h("span", { style: { width: `${pct}%` } }),
      threshold ? h("span", { class: "threshold", style: { left: `${threshold * 100}%` } }) : null),
    fmtScore(value));
}

function badge(text, kind) { return h("span", { class: `badge ${kind || ""}`, text }); }

/* ---------------------------------------------------------------- routing */

const ROUTES = { "": "home", "#/": "home", "#/ask": "ask", "#/library": "library", "#/visual": "visual", "#/evaluation": "evaluation" };

function route() {
  const view = ROUTES[location.hash] || "home";
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.dataset.view !== view; });
  document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.route === view));
  if (view === "library") loadDocuments();
  if (view === "evaluation") loadEvaluation();
  if (view === "home") { renderHistory(); loadHealth(); }
  if (view === "ask") setTimeout(() => $("#chatInput").focus(), 0);
}

/* ---------------------------------------------------------------- status */

async function loadInfo() {
  try {
    state.info = await api.get("/system/info");
    const notice = $("#notice");
    const parts = [];
    if (!state.info.authentication) parts.push("Public demo without authentication: anything you upload is visible to every visitor. Do not upload private or sensitive documents.");
    parts.push(state.info.storage_notice);
    if (!state.info.image_search_enabled) parts.push("Image search is disabled on this deployment.");
    notice.textContent = parts.join(" ");
    notice.hidden = false;
    $("#uploadHint").textContent = `PDF, Markdown, TXT, PNG or JPEG · up to ${fmtBytes(state.info.max_upload_size_bytes)}`;
    $("#libraryNotice").textContent = state.info.storage_notice;
    $("#statStorage").textContent = state.info.storage_mode === "persistent" ? "Persistent" : "Ephemeral demo";
  } catch (err) { /* status pill reports connectivity */ }
}

async function loadReady() {
  const pill = $("#statusPill");
  try {
    const res = await fetch("/ready");
    const data = await res.json();
    state.ready = data;
    pill.className = "status-pill " + (data.ready ? "ok" : "warn");
    $("#statusText").textContent = data.ready ? `Ready · ${data.components.text_index.detail}` : `Starting · ${data.phase}`;
    renderStatusList(data);
    if (!data.ready) setTimeout(loadReady, 4000);
  } catch (err) {
    pill.className = "status-pill err";
    $("#statusText").textContent = "Offline";
  }
}

function renderStatusList(data) {
  const list = clear($("#statusList"));
  const labels = { storage: "Storage", registry: "Document registry", text_index: "Text index", image_index: "Image index", text_model: "Text model (MiniLM)", image_model: "Image model (CLIP)" };
  for (const [key, label] of Object.entries(labels)) {
    const c = data.components[key];
    if (!c) continue;
    list.append(h("li", {}, h("span", {}, label), h("span", { class: "muted", title: c.detail }, badge(c.ready ? "ready" : "pending", c.ready ? "ok" : "warn"), " ", c.detail)));
  }
  if (typeof data.memory_rss_mb === "number") list.append(h("li", {}, h("span", {}, "Process memory"), h("span", { class: "muted" }, `${data.memory_rss_mb} MB`)));
}

async function loadHealth() {
  try {
    const hdata = await api.get("/health");
    $("#statDocs").textContent = hdata.indexed_documents;
    $("#statChunks").textContent = hdata.indexed_text_chunks;
    $("#statImages").textContent = hdata.indexed_images;
  } catch (_) { /* ignore */ }
}

/* ---------------------------------------------------------------- home */

function renderExamples() {
  const box = clear($("#examples"));
  const chips = clear($("#suggestions"));
  for (const ex of EXAMPLES) {
    box.append(h("button", { class: "example", type: "button", on: { click: () => ask(ex.q) } }, h("span", { class: "tag", text: ex.tag }), h("span", { text: ex.q })));
    chips.append(h("button", { class: "chip", type: "button", on: { click: () => ask(ex.q) } }, ex.q.length > 58 ? ex.q.slice(0, 55) + "…" : ex.q));
  }
}

function getHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch (_) { return []; }
}
function pushHistory(q, type) {
  const items = getHistory().filter((i) => i.q !== q);
  items.unshift({ q, type, at: new Date().toISOString() });
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 12))); } catch (_) { /* storage disabled */ }
}
function renderHistory() {
  const list = clear($("#historyList"));
  const items = getHistory();
  if (!items.length) { list.append(h("li", { class: "muted small", text: "Questions you ask are kept in this browser only." })); return; }
  for (const item of items) {
    list.append(h("li", {}, h("button", { type: "button", on: { click: () => ask(item.q) } }, h("span", { text: item.q }), h("span", { class: "muted small", text: fmtDate(item.at) }))));
  }
}

/* ---------------------------------------------------------------- ask */

function ask(query) {
  location.hash = "#/ask";
  $("#chatInput").value = query;
  sendChat();
}

function addMessage(role, ...content) {
  const log = $("#chatLog");
  const msg = h("div", { class: `msg ${role}` }, ...content);
  log.append(msg);
  log.scrollTop = log.scrollHeight;
  return msg;
}

function renderAnswerText(answer, citations) {
  const bubble = h("div", { class: "bubble" });
  for (const para of answer.split(/\n{2,}/)) {
    const p = h("p");
    let last = 0;
    CITATION_RE.lastIndex = 0;
    let m;
    while ((m = CITATION_RE.exec(para)) !== null) {
      p.append(para.slice(last, m.index));
      if (m[1]) {
        const filename = m[1].trim();
        const page = m[2];
        p.append(h("button", { class: "cite", type: "button", title: "Show the cited passage", on: { click: () => focusPassage(filename, page) } }, `${filename} · p.${page}`));
      } else {
        p.append(h("span", { class: "cite removed", title: "A citation that could not be verified was removed" }, "citation removed"));
      }
      last = m.index + m[0].length;
    }
    p.append(para.slice(last));
    bubble.append(p);
  }
  return bubble;
}

async function sendChat() {
  const input = $("#chatInput");
  const query = input.value.trim();
  if (!query || state.busy) return;
  state.busy = true;
  $("#chatSend").disabled = true;
  input.value = "";
  $("#suggestions").hidden = true;
  addMessage("user", h("div", { class: "bubble", text: query }));
  const pending = addMessage("bot", h("div", { class: "bubble" }, h("span", { class: "thinking" }, h("i"), h("i"), h("i"), " Retrieving evidence and verifying citations…")));
  try {
    const data = await api.post("/chat", { query, include_debug: true });
    pending.remove();
    renderBotAnswer(query, data);
    pushHistory(query, data.answer_type);
  } catch (err) {
    pending.remove();
    addMessage("bot",
      h("div", { class: "bubble" }, h("strong", { text: "Something went wrong. " }), err.message),
      h("div", { class: "meta-row" }, h("button", { class: "btn ghost small", type: "button", on: { click: (e) => { e.target.closest(".msg").remove(); input.value = query; sendChat(); } } }, "Retry")));
  } finally {
    state.busy = false;
    $("#chatSend").disabled = false;
  }
}

function renderBotAnswer(query, data) {
  state.last = data;
  const typeBadge = {
    answer: badge("Grounded answer", "ok"),
    visual_only: badge("Visual evidence only", "accent"),
    insufficient_evidence: badge("Insufficient evidence", "warn"),
  }[data.answer_type];
  const check = data.citation_check || {};
  const meta = h("div", { class: "meta-row" }, typeBadge,
    check.total ? badge(`${check.verified}/${check.total} citations verified`, check.all_verified ? "ok" : "warn") : null,
    h("span", { title: "Similarity of the best text passage (MiniLM cosine). The answer gate is shown in the inspector." }, `top passage ${fmtScore(data.debug_info ? data.debug_info.top_text_score : data.confidence)}`),
    data.latency_ms ? h("span", { text: `${Math.round(data.latency_ms)} ms` }) : null,
    h("span", { title: data.provider === "extractive" ? "Answer quoted from the retrieved passages; no LLM involved" : "Answer written by the LangChain tool-calling agent and verified against the retrieved passages" },
      data.provider === "extractive" ? "extractive (no LLM)" : data.provider),
    data.pii_redactions ? badge(`${data.pii_redactions} PII value${data.pii_redactions > 1 ? "s" : ""} masked`, "accent") : null,
    h("button", { class: "link-btn", type: "button", on: { click: () => renderInspector(data) } }, "Inspect evidence"));
  const parts = [renderAnswerText(data.answer, data.citations), meta];
  if (data.agent_trace && data.agent_trace.length) {
    parts.push(h("div", { class: "agent-trace", "aria-label": "Agent steps" },
      h("span", { class: "muted small", text: "Agent steps:" }),
      data.agent_trace.map((s, i) => h("span", { class: "step", title: `${s.results} result(s) in ${Math.round(s.duration_ms)} ms` },
        `${i + 1}. ${s.tool === "search_diagrams" ? "🖼" : "🔎"} ${s.tool}("${s.query}") → ${s.results}`))));
  }
  if (data.warnings && data.warnings.length) parts.push(h("ul", { class: "warnings" }, data.warnings.map((w) => h("li", { text: w }))));
  if (data.images && data.images.length) parts.push(h("div", { class: "thumbs" }, data.images.map((img) => imageCard(img.image_url, img.filename, img.page_number, img.score, img.match_reason, img.doc_id, img.description))));
  addMessage("bot", ...parts);
  renderInspector(data);
}

function renderInspector(data) {
  const box = clear($("#inspector"));
  const threshold = state.info ? state.info.thresholds.text_min_score : null;
  box.append(h("p", { class: "muted small" },
    "Passages ranked by MiniLM cosine similarity. ",
    threshold ? `An answer requires at least one passage ≥ ${fmtScore(threshold)} (yellow marker). ` : "",
    "Cited passages are outlined."));
  if (!data.passages || !data.passages.length) box.append(h("p", { class: "muted", text: "No passages were retrieved." }));
  for (const p of data.passages || []) {
    const cls = "passage" + (p.cited ? " cited" : "") + (p.accepted ? "" : " dim");
    box.append(h("div", { class: cls, dataset: { file: p.filename, page: String(p.page_number) } },
      h("div", { class: "passage-head" },
        h("button", { class: "passage-src", type: "button", title: "Open document", on: { click: () => openDocument(p.doc_id) } }, `${p.filename} · page ${p.page_number}`),
        h("span", { class: "meta-row" }, p.cited ? badge("cited", "ok") : p.accepted ? badge("used", "accent") : badge("below threshold"), scoreBar(p.score, threshold))),
      p.section ? h("div", { class: "section-path", text: p.section }) : null,
      h("div", { class: "passage-text", text: p.text })));
  }
  if (data.images && data.images.length) {
    box.append(h("h3", { class: "small muted", text: "Visual evidence (CLIP)" }));
    box.append(h("div", { class: "thumbs" }, data.images.map((img) => imageCard(img.image_url, img.filename, img.page_number, img.score, img.match_reason, img.doc_id))));
  }
}

function focusPassage(filename, page) {
  if (state.last) renderInspector(state.last);
  const el = [...document.querySelectorAll("#inspector .passage")].find((p) => p.dataset.file === filename && p.dataset.page === String(page));
  if (!el) { toast("That passage is not in the current evidence list."); return; }
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
}

/* ---------------------------------------------------------------- images */

function imageCard(url, filename, page, score, reason, docId, caption) {
  return h("div", { class: "img-card" },
    h("button", { class: "img-btn", type: "button", title: "Enlarge", on: { click: () => openLightbox(url, `${filename}${page ? ` · page ${page}` : ""}`) } },
      h("img", { src: url, alt: `Image from ${filename}${page ? `, page ${page}` : ""}`, loading: "lazy" })),
    h("div", { class: "img-meta" },
      h("span", { class: "name", title: caption || filename, text: filename }),
      caption ? h("span", { class: "caption muted small", text: caption }) : null,
      h("span", { class: "muted" }, page ? `Page ${page}` : "Standalone image"),
      typeof score === "number" ? scoreBar(score) : null,
      typeof score === "number" ? h("span", { class: "muted small", text: reason ? (REASONS[reason] || reason) : "Below the relevance thresholds" }) : null,
      docId ? h("button", { class: "link-btn", type: "button", on: { click: () => openDocument(docId) } }, "Open source document") : null));
}

function openLightbox(url, caption) {
  $("#lightboxImg").src = url;
  $("#lightboxImg").alt = caption;
  $("#lightboxCaption").textContent = caption;
  $("#lightbox").hidden = false;
  $("#lightboxClose").focus();
}

/* ---------------------------------------------------------------- library */

async function loadDocuments() {
  const rows = clear($("#docRows"));
  rows.append(h("tr", {}, h("td", { colspan: "8" }, h("div", { class: "skeleton" }))));
  try {
    const data = await api.get("/documents");
    state.docs = data.documents;
    renderDocuments();
  } catch (err) {
    clear(rows).append(h("tr", {}, h("td", { colspan: "8", class: "muted", text: err.message })));
  }
}

function renderDocuments() {
  const q = $("#docSearch").value.trim().toLowerCase();
  const type = $("#docType").value;
  const status = $("#docStatus").value;
  const rows = clear($("#docRows"));
  const docs = state.docs.filter((d) => {
    if (q && !d.filename.toLowerCase().includes(q)) return false;
    if (status && d.status !== status) return false;
    if (type === "image" && !["png", "jpg", "jpeg"].includes(d.file_type)) return false;
    if (type && type !== "image" && d.file_type !== type) return false;
    return true;
  }).slice().reverse();
  $("#docEmpty").hidden = docs.length > 0;
  for (const d of docs) {
    const statusBadge = d.status === "ready" ? badge(d.warnings && d.warnings.length ? "ready · warnings" : "ready", d.warnings && d.warnings.length ? "warn" : "ok")
      : d.status === "failed" ? badge("failed", "err") : badge(d.status);
    rows.append(h("tr", {},
      h("td", {}, h("button", { class: "doc-name", type: "button", on: { click: () => openDocument(d.doc_id) } }, d.filename), d.source === "sample" ? h("span", {}, " ", badge("sample")) : null),
      h("td", {}, typeLabel(d.file_type)),
      h("td", { title: d.error || (d.warnings || []).join(" ") }, statusBadge),
      h("td", { class: "num" }, d.status === "ready" ? d.page_count : "–"),
      h("td", { class: "num" }, d.chunk_count),
      h("td", { class: "num" }, d.image_count),
      h("td", { class: "muted" }, fmtDate(d.created_at)),
      h("td", {}, h("div", { class: "actions" },
        h("button", { class: "btn ghost small", type: "button", on: { click: () => openDocument(d.doc_id) } }, "Details"),
        h("button", { class: "btn danger small", type: "button", "aria-label": `Delete ${d.filename}`, on: { click: () => deleteDocument(d) } }, "Delete")))));
  }
}

async function deleteDocument(d) {
  if (!confirm(`Delete "${d.filename}" and all of its passages and images?`)) return;
  try {
    await api.del(`/documents/${encodeURIComponent(d.doc_id)}`);
    toast(`Deleted ${d.filename}`);
    closeDrawer();
    loadDocuments();
    loadHealth();
  } catch (err) { toast(err.message, true); }
}

async function uploadFile(file) {
  const status = clear($("#uploadStatus"));
  if (!file) return;
  if (state.info && file.size > state.info.max_upload_size_bytes) {
    status.append(badge("too large", "err"), ` ${file.name} is ${fmtBytes(file.size)}; the limit is ${fmtBytes(state.info.max_upload_size_bytes)}.`);
    return;
  }
  status.append(h("span", { class: "thinking" }, h("i"), h("i"), h("i"), ` Processing ${file.name}: extracting, embedding and indexing…`));
  const form = new FormData();
  form.append("file", file);
  try {
    const data = await api.post("/documents/upload", form);
    clear(status).append(badge("indexed", "ok"), ` ${data.document.filename}: ${data.chunks_indexed} passages, ${data.images_indexed} images, ${data.document.page_count} page(s).`);
    if (data.warnings && data.warnings.length) status.append(h("ul", { class: "warnings" }, data.warnings.map((w) => h("li", { text: w }))));
    toast(`Indexed ${data.document.filename}`);
  } catch (err) {
    const existing = err.body && err.body.detail && err.body.detail.existing_document;
    clear(status).append(badge(err.status === 409 ? "duplicate" : "failed", err.status === 409 ? "warn" : "err"), " ", existing ? `Already in the library as "${existing.filename}".` : err.message);
  } finally {
    $("#fileInput").value = "";
    loadDocuments();
    loadHealth();
  }
}

/* ---------------------------------------------------------------- drawer */

async function openDocument(docId) {
  const body = clear($("#drawerBody"));
  $("#drawerTitle").textContent = "Loading…";
  $("#drawer").hidden = false;
  $("#drawerBackdrop").hidden = false;
  $("#drawerClose").focus();
  body.append(h("div", { class: "skeleton" }), h("div", { class: "skeleton" }));
  try {
    const data = await api.get(`/documents/${encodeURIComponent(docId)}`);
    const d = data.document;
    $("#drawerTitle").textContent = d.filename;
    clear(body);
    body.append(h("dl", { class: "kv" },
      h("dt", { text: "Status" }), h("dd", {}, badge(d.status, d.status === "ready" ? "ok" : d.status === "failed" ? "err" : "")),
      h("dt", { text: "Type" }), h("dd", { text: `${typeLabel(d.file_type)} · ${fmtBytes(d.file_size_bytes)}` }),
      h("dt", { text: "Pages" }), h("dd", { text: d.pages_with_text != null && d.file_type === "pdf" ? `${d.page_count} (${d.pages_with_text} with text)` : String(d.page_count) }),
      h("dt", { text: "Passages / images" }), h("dd", { text: `${d.chunk_count} / ${d.image_count}` }),
      h("dt", { text: "Added" }), h("dd", { text: fmtDate(d.created_at) }),
      h("dt", { text: "Source" }), h("dd", { text: d.source === "sample" ? "Bundled sample" : "Upload" }),
      h("dt", { text: "Document ID" }), h("dd", {}, h("code", { text: d.doc_id })),
      h("dt", { text: "SHA-256" }), h("dd", {}, h("code", { title: d.content_hash, text: `${d.content_hash.slice(0, 16)}…` }))));
    if (d.error) body.append(h("ul", { class: "warnings" }, h("li", { text: d.error })));
    if (d.warnings && d.warnings.length) body.append(h("ul", { class: "warnings" }, d.warnings.map((w) => h("li", { text: w }))));
    if (data.images.length) {
      body.append(h("h3", { class: "small muted", text: "Images" }),
        h("div", { class: "thumbs" }, data.images.map((img) => imageCard(img.url_path, img.filename, img.page_number, undefined, null, null))));
    }
    const byPage = new Map();
    for (const c of data.chunks) { if (!byPage.has(c.page_number)) byPage.set(c.page_number, []); byPage.get(c.page_number).push(c); }
    for (const [page, chunks] of byPage) {
      body.append(h("div", { class: "page-group" }, h("h3", { text: `Page ${page} · ${chunks.length} passage${chunks.length > 1 ? "s" : ""}` }),
        ...chunks.map((c) => h("div", { class: "passage" }, c.section ? h("div", { class: "section-path", text: c.section }) : null, h("div", { class: "passage-text", text: c.text })))));
    }
    if (d.status === "ready") body.append(h("div", {}, h("button", { class: "btn danger small", type: "button", on: { click: () => deleteDocument(d) } }, "Delete document")));
  } catch (err) {
    $("#drawerTitle").textContent = "Document";
    clear(body).append(h("p", { class: "muted", text: err.message }));
  }
}

function closeDrawer() { $("#drawer").hidden = true; $("#drawerBackdrop").hidden = true; }

/* ---------------------------------------------------------------- visual search */

async function visualSearch(form, label) {
  const box = clear($("#visualResults"));
  if (state.info && !state.info.image_search_enabled) { box.append(h("p", { class: "muted", text: "Image search is disabled on this deployment." })); return; }
  box.append(h("p", { class: "muted" }, h("span", { class: "thinking" }, h("i"), h("i"), h("i"), ` Searching images for ${label}…`)));
  try {
    const results = await api.post("/search/images", form);
    clear(box);
    if (!results.length) { box.append(h("p", { class: "muted", text: "No images are indexed yet. Upload a PNG/JPEG or a PDF with figures." })); return; }
    for (const r of results) box.append(imageCard(r.image.url_path, r.image.filename, r.image.page_number, r.score, r.match_reason, r.image.doc_id, r.image.caption));
  } catch (err) {
    clear(box).append(h("p", { class: "muted", text: err.message }));
  }
}

/* ---------------------------------------------------------------- evaluation */

const METRICS = [
  ["text_recall_at_k", "Text Recall@5", "Relevant file+page in the top-5 passages", "pct"],
  ["groundedness", "Groundedness", "Answer claims supported by their cited passage", "pct"],
  ["text_mrr", "Text MRR", "Mean reciprocal rank of the first relevant passage", "num"],
  ["text_precision_at_k", "Text Precision@K", "Share of top-K passages that are relevant", "pct"],
  ["image_recall_at_k", "Image Recall@3", "Relevant diagram in the top-3 (CLIP + captions)", "pct"],
  ["image_mrr", "Image MRR", "Rank of the first relevant image", "num"],
  ["citation_precision", "Citation precision", "Cited file+page pairs that are relevant", "pct"],
  ["citation_verification_rate", "Citations verified", "Citation tags that passed the verifier", "pct"],
  ["cited_claim_lexical_support", "Claim support", "Word overlap of cited claims with passages", "pct"],
  ["fact_coverage", "Fact coverage", "Expected facts present in the answers", "pct"],
  ["answer_rate", "Answer rate", "Answerable questions not refused", "pct"],
  ["negative_rejection_rate", "Negative rejection", "Out-of-corpus questions refused", "pct"],
  ["hard_negative_handling_rate", "Hard negatives", "Near-domain questions refused or flagged", "pct"],
  ["latency_p50_ms", "Latency p50", "Retrieve + answer, warm models", "ms"],
  ["latency_p95_ms", "Latency p95", "Retrieve + answer, warm models", "ms"],
];

async function loadEvaluation() {
  const body = clear($("#evalBody"));
  body.append(h("div", { class: "skeleton" }));
  try {
    const r = await api.get("/evaluation");
    clear(body);
    const e = r.environment || {};
    const d = r.dataset || {};
    $("#evalMeta").textContent = `Report generated ${fmtDate(r.timestamp)} · ${d.size} queries (${d.text_queries} text, ${d.visual_queries} visual, ${d.negative_queries} negative, ${d.hard_negative_queries} hard negative) · provider ${e.provider} · K=${e.k_text}/${e.k_images}`;
    const metrics = h("div", { class: "metrics" });
    for (const [key, label, desc, kind] of METRICS) {
      const v = r.summary[key];
      const text = v === null || v === undefined ? "n/a" : kind === "pct" ? fmtPct(v) : kind === "ms" ? `${v.toFixed(1)} ms` : v.toFixed(3);
      metrics.append(h("div", { class: "metric" }, h("div", { class: "v", text }), h("div", { class: "l", text: label }), h("div", { class: "d", text: desc })));
    }
    body.append(metrics);
    if (r.security) {
      const s = r.security.summary;
      const sec = h("div", { class: "metrics" });
      for (const [label, value, desc] of [
        ["Prompt-injection resistance", fmtPct(s.prompt_injection_resistance), `${r.security.dataset.prompt_injection} attacks via a poisoned bulletin and the question itself`],
        ["PII-leak rate", fmtPct(s.pii_leak_rate), `${r.security.dataset.pii} probes for planted e-mails, phones, SSNs, cards, VINs, names`],
        ["PII values masked", String(s.pii_values_masked), "Across all security responses"],
      ]) sec.append(h("div", { class: "metric" }, h("div", { class: "v", text: value }), h("div", { class: "l", text: label }), h("div", { class: "d", text: desc })));
      body.append(h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", { text: "Security suite" })), sec));
    }
    const c = r.comparison || {};
    body.append(h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", { text: "Hybrid ranking vs. baseline (document MRR)" })),
      h("p", { class: "muted small", text: `Weighted Reciprocal Rank Fusion: ${c.hybrid_rrf_document_mrr ?? "n/a"} · raw-score mixing of MiniLM and CLIP: ${c.raw_score_mixing_document_mrr ?? "n/a"}` }),
      h("p", { class: "muted small", text: "The dataset is small and authored alongside the sample corpus; treat the numbers as a regression check, not a general benchmark." })));
    const table = h("table", { class: "eval-table" }, h("thead", {}, h("tr", {}, ["Query", "Type", "Top text / image", "RR", "Grounded", "Citation precision", "Answer", "ms", "Result"].map((t) => h("th", { text: t })))));
    const tbody = h("tbody");
    for (const q of r.query_results) {
      const kind = q.is_unsupported ? (q.hard_negative ? "hard negative" : "negative") : q.modality;
      tbody.append(h("tr", {},
        h("td", { title: q.generated_answer, text: q.query }),
        h("td", { text: kind }),
        h("td", { text: `${fmtScore(q.top_text_score)} / ${fmtScore(q.top_image_score)}` }),
        h("td", { text: q.reciprocal_rank == null ? "–" : q.reciprocal_rank.toFixed(2) }),
        h("td", { text: q.groundedness == null ? "–" : fmtPct(q.groundedness) }),
        h("td", { text: q.citation_precision == null ? "–" : fmtPct(q.citation_precision) }),
        h("td", { text: q.answer_type }),
        h("td", { text: Math.round(q.latency_ms) }),
        h("td", {}, badge(q.passed ? "pass" : "fail", q.passed ? "ok" : "err"))));
    }
    table.append(tbody);
    body.append(h("div", { class: "card table-card" }, table));
  } catch (err) {
    clear(body).append(h("div", { class: "card" }, h("h2", { text: "No evaluation report yet" }),
      h("p", { class: "muted" }, err.status === 404 ? "Generate one locally with " : err.message, err.status === 404 ? h("code", { text: "python scripts/run_eval.py" }) : null, err.status === 404 ? " and commit reports/evaluation_report.json." : "")));
    $("#evalMeta").textContent = "";
  }
}

/* ---------------------------------------------------------------- wiring */

function wire() {
  window.addEventListener("hashchange", route);
  $("#chatForm").addEventListener("submit", (e) => { e.preventDefault(); sendChat(); });
  $("#chatInput").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } });
  $("#chatInput").addEventListener("input", (e) => { e.target.style.height = "auto"; e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`; });
  $("#refreshStatus").addEventListener("click", () => { loadReady(); loadHealth(); });
  $("#statusPill").addEventListener("click", () => { location.hash = "#/"; loadReady(); });
  $("#clearHistory").addEventListener("click", () => { localStorage.removeItem(HISTORY_KEY); renderHistory(); });
  document.querySelector("[data-action=open-upload]").addEventListener("click", () => setTimeout(() => $("#dropzone").focus(), 50));

  const dz = $("#dropzone");
  dz.tabIndex = 0;
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#fileInput").click(); } });
  $("#fileInput").addEventListener("change", (e) => uploadFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => uploadFile(e.dataTransfer.files[0]));
  ["#docSearch", "#docType", "#docStatus"].forEach((s) => $(s).addEventListener("input", renderDocuments));

  $("#visualTextForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("#visualQuery").value.trim();
    if (!q) return;
    const form = new FormData(); form.append("query", q); form.append("top_k", "6");
    visualSearch(form, `“${q}”`);
  });
  let visualFile = null;
  $("#visualFile").addEventListener("change", (e) => {
    visualFile = e.target.files[0] || null;
    $("#visualImageBtn").disabled = !visualFile;
    const preview = $("#visualPreview");
    if (visualFile) {
      if (preview.src) URL.revokeObjectURL(preview.src);
      preview.src = URL.createObjectURL(visualFile);
      preview.hidden = false;
      $("#visualDropText").textContent = visualFile.name;
    }
  });
  $("#visualImageForm").addEventListener("submit", (e) => {
    e.preventDefault();
    if (!visualFile) return;
    const form = new FormData(); form.append("image_file", visualFile); form.append("top_k", "6");
    visualSearch(form, visualFile.name);
  });

  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#drawerBackdrop").addEventListener("click", closeDrawer);
  $("#lightboxClose").addEventListener("click", () => { $("#lightbox").hidden = true; });
  $("#lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") $("#lightbox").hidden = true; });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!$("#lightbox").hidden) $("#lightbox").hidden = true;
    else if (!$("#drawer").hidden) closeDrawer();
  });

  addMessage("bot", h("div", { class: "bubble" },
    h("p", {}, h("strong", { text: "Ask anything about the indexed documents." })),
    h("p", { text: "Answers quote retrieved passages and every [Source, Page] citation is verified against the document. If the documents don't contain the answer, I'll say so instead of guessing." })));
}

document.addEventListener("DOMContentLoaded", () => {
  wire();
  renderExamples();
  loadInfo();
  loadReady();
  route();
  setInterval(() => { if (!document.hidden) { loadReady(); loadHealth(); } }, 30000);
});
