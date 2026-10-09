// Browser glue for the static demo: fetch manifest + index, load the ONNX embedder
// with transformers.js, describe an uploaded image with a small vision model in a
// Web Worker (or take the cached description of an example), embed the description,
// rank with scorer.js, and draw the merged paths of the results as a tree.
// Everything stays in the visitor's browser; no request carries the image anywhere.
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";
import {
  AUTO_LEVEL,
  LEVELS,
  buildIndex,
  chainOf,
  formatCode,
  normalizeQuery,
  parseVectors,
  rank,
} from "./scorer.js";

env.allowLocalModels = false;

const $ = (id) => document.getElementById(id);
const ui = {
  drop: $("drop"),
  dropText: $("drop-text"),
  preview: $("preview"),
  file: $("file"),
  gallery: $("gallery"),
  text: $("text"),
  descNote: $("desc-note"),
  describe: $("describe"),
  level: $("level"),
  topK: $("topk"),
  noColours: $("nocolours"),
  principal: $("principal"),
  run: $("run"),
  status: $("status"),
  graph: $("graph"),
  legend: $("legend"),
  tip: $("tip"),
  results: $("results"),
  meta: $("meta"),
};

const state = {
  manifest: null,
  examples: [],
  index: null,
  indexLoad: null,
  embedder: null,
  embedderLoad: null,
  visionWorker: null,
  visionRequests: new Map(), // request id -> {resolve, reject}
  visionSeq: 0,
  visionReady: false,
  progress: new Map(), // label -> {loaded, total, unit}
  pending: 0,
  error: null,
  image: null, // {bytes: ArrayBuffer, mime, name}
  example: null, // the gallery example shown, with its cached description
};

// -- status -------------------------------------------------------------------

function fmtMB(bytes) {
  return `${(bytes / 1e6).toFixed(bytes < 10e6 ? 1 : 0)} MB`;
}

function fmtProgress({ loaded, total, unit }) {
  if (unit === "tokens") return `${loaded} tokens`;
  return total ? `${fmtMB(loaded)} / ${fmtMB(total)}` : fmtMB(loaded);
}

function renderStatus(message) {
  ui.status.innerHTML = "";
  if (state.error) {
    ui.status.className = "status error";
    ui.status.textContent = state.error;
    return;
  }
  ui.status.className = "status";
  if (!state.progress.size) {
    ui.status.textContent =
      message || (state.pending ? "Loading…" : "Ready. Everything runs in this browser tab.");
    return;
  }
  for (const [label, entry] of state.progress) {
    const { loaded, total } = entry;
    const row = document.createElement("div");
    row.className = "progress";
    const pct = total ? Math.min(100, Math.round((100 * loaded) / total)) : 0;
    row.innerHTML = `<span></span><div class="bar"><i style="width:${pct}%"></i></div><small></small>`;
    row.querySelector("span").textContent = label;
    row.querySelector("small").textContent = fmtProgress(entry);
    ui.status.appendChild(row);
  }
}

function setProgress(label, loaded, total, unit) {
  state.progress.set(label, { loaded, total, unit });
  renderStatus();
}

function doneProgress(label) {
  state.progress.delete(label);
  renderStatus();
}

function fail(err, prefix = "Something failed") {
  state.error = `${prefix}: ${err.message || err}`;
  renderStatus();
  console.error(err);
}

// -- loading ------------------------------------------------------------------

async function fetchWithProgress(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    setProgress(label, loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  doneProgress(label);
  return out.buffer;
}

function loadIndex() {
  if (state.index) return Promise.resolve(state.index);
  if (!state.indexLoad) {
    state.pending += 1;
    state.indexLoad = fetchIndex()
      .catch((err) => {
        state.indexLoad = null;
        throw err;
      })
      .finally(() => {
        state.pending -= 1;
        renderStatus();
      });
  }
  return state.indexLoad;
}

async function fetchIndex() {
  const entry = state.manifest.index;
  const label = `Vienna ${entry.edition} ${entry.lang}`;
  const [schemeBuf, vecBuf] = await Promise.all([
    fetchWithProgress(entry.scheme, `${label} scheme`),
    fetchWithProgress(entry.vectors, `${label} vectors`),
  ]);
  const scheme = JSON.parse(new TextDecoder().decode(schemeBuf));
  const vectors = parseVectors(vecBuf, {
    rows: entry.rows,
    dim: state.manifest.dim,
    encoding: entry.encoding,
  });
  state.index = buildIndex(scheme, vectors);
  return state.index;
}

function loadEmbedder() {
  if (state.embedder) return Promise.resolve(state.embedder);
  if (!state.embedderLoad) {
    const { web_model: model, web_dtype: dtype } = state.manifest;
    const label = `embedder ${model} (${dtype})`;
    state.pending += 1;
    state.embedderLoad = pipeline("feature-extraction", model, {
      dtype,
      progress_callback: (e) => {
        if (e.status === "progress" && e.file && e.file.endsWith(".onnx")) {
          setProgress(label, e.loaded, e.total);
        }
      },
    })
      .then((p) => {
        state.embedder = p;
        return p;
      })
      .catch((err) => {
        state.embedderLoad = null;
        throw err;
      })
      .finally(() => {
        state.pending -= 1;
        doneProgress(label);
      });
  }
  return state.embedderLoad;
}

// -- vision model in a Web Worker ---------------------------------------------

function visionWorker() {
  if (!state.visionWorker) {
    const worker = new Worker("vision-worker.js", { type: "module" });
    const { web_model: model } = state.manifest.vision;
    const files = new Map(); // file -> {loaded, total}
    const loadLabel = `vision model ${model}`;
    const genLabel = "describing";
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") {
        files.set(m.file, { loaded: m.loaded, total: m.total });
        let loaded = 0;
        let total = 0;
        for (const f of files.values()) {
          loaded += f.loaded;
          total += f.total;
        }
        setProgress(loadLabel, loaded, total);
      } else if (m.type === "ready") {
        state.visionReady = true;
        state.visionDevice = m.device;
        doneProgress(loadLabel);
      } else if (m.type === "fallback") {
        files.clear();
        doneProgress(loadLabel);
        renderStatus(`Vision model: ${m.device} failed (${m.message}); trying the next option…`);
      } else if (m.type === "generating") {
        setProgress(genLabel, m.tokens, state.manifest.vision.max_new_tokens, "tokens");
      } else if (m.id !== undefined && state.visionRequests.has(m.id)) {
        const { resolve, reject } = state.visionRequests.get(m.id);
        state.visionRequests.delete(m.id);
        doneProgress(loadLabel);
        doneProgress(genLabel);
        if (m.type === "error") reject(new Error(m.message));
        else resolve(m);
      }
    };
    worker.onerror = (e) => {
      for (const { reject } of state.visionRequests.values()) {
        reject(new Error(e.message || "vision worker failed"));
      }
      state.visionRequests.clear();
      doneProgress(loadLabel);
      doneProgress(genLabel);
    };
    state.visionWorker = worker;
  }
  return state.visionWorker;
}

/** WebGPU, and whether its adapter runs fp16 shaders (needed for the fp16 weights). */
async function gpuSupport() {
  if (!navigator.gpu) return { device: "wasm", f16: false };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { device: "wasm", f16: false };
    return { device: "webgpu", f16: adapter.features.has("shader-f16") };
  } catch {
    return { device: "wasm", f16: false };
  }
}

async function describeInBrowser(image) {
  const v = state.manifest.vision;
  const worker = visionWorker();
  const id = ++state.visionSeq;
  const { device, f16 } = await gpuSupport();
  return new Promise((resolve, reject) => {
    state.visionRequests.set(id, { resolve, reject });
    worker.postMessage({
      id,
      type: "describe",
      modelId: v.web_model,
      kind: v.kind || "chat",
      dtype: v.dtype,
      device,
      f16,
      bytes: image.bytes.slice(0), // a copy: the original stays usable for a second run
      mime: image.mime,
      prompt: v.prompt,
      maxNewTokens: v.max_new_tokens,
    });
  });
}

let describing = false;

async function describeCurrent() {
  if (!state.image || describing || !state.manifest.vision) return;
  describing = true;
  ui.describe.disabled = true;
  state.error = null;
  try {
    renderStatus(state.visionReady ? "Describing…" : "Loading the vision model…");
    const t0 = performance.now();
    const { text, device } = await describeInBrowser(state.image);
    ui.text.value = text;
    ui.descNote.textContent =
      `Written by ${state.manifest.vision.name} in this browser just now ` +
      `(${device}, ${Math.round((performance.now() - t0) / 100) / 10} s). Edit what it missed, ` +
      "or write your own.";
    renderStatus();
  } catch (err) {
    fail(err, "The vision model failed");
  } finally {
    describing = false;
    ui.describe.disabled = !state.image; // the note written above stays
  }
}

/** The Describe button and the note follow whether an image, an example, is shown. */
function updateDescribeUi() {
  const v = state.manifest.vision;
  if (!v) {
    ui.describe.hidden = true;
    return;
  }
  ui.describe.disabled = !state.image;
  ui.describe.title = `${v.web_model} runs in a Web Worker in this tab; about 250 MB once`;
  if (state.example) {
    const ex = state.example;
    ui.text.value = ex.description || "";
    ui.descNote.textContent = ex.description
      ? `Written offline by ${ex.model}, the same model this page runs (click Describe to run ` +
        "it here; quantised weights may word it slightly differently). Editable."
      : "No cached description for this example: click Describe.";
  } else if (state.image) {
    ui.descNote.textContent = "Click Describe to let the vision model write the inventory here, or type it.";
  }
}

// -- image input --------------------------------------------------------------

function setImage(bytes, mime, name, example = null) {
  state.image = { bytes, mime, name };
  state.example = example;
  const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
  ui.preview.src = url;
  ui.preview.alt = name;
  ui.preview.hidden = false;
  ui.dropText.textContent = name;
  if (!example) ui.text.value = "";
  state.error = null;
  updateDescribeUi();
  renderStatus();
}

async function takeFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const bytes = await file.arrayBuffer();
  selectGallery(null);
  setImage(bytes, file.type, file.name);
}

function bindImageInput() {
  ui.file.addEventListener("change", () => takeFile(ui.file.files[0]));
  ui.drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    ui.drop.classList.add("over");
  });
  ui.drop.addEventListener("dragleave", () => ui.drop.classList.remove("over"));
  ui.drop.addEventListener("drop", (e) => {
    e.preventDefault();
    ui.drop.classList.remove("over");
    takeFile(e.dataTransfer.files[0]);
  });
  document.addEventListener("paste", (e) => {
    const item = Array.from(e.clipboardData?.items || []).find((i) => i.type.startsWith("image/"));
    if (item) takeFile(item.getAsFile());
  });
}

// -- examples -----------------------------------------------------------------

async function loadExamples() {
  try {
    const res = await fetch("examples.json");
    if (res.ok) state.examples = await res.json();
  } catch {
    state.examples = [];
  }
}

function selectGallery(button) {
  for (const b of ui.gallery.querySelectorAll("button")) b.classList.toggle("selected", b === button);
}

function renderGallery() {
  ui.gallery.innerHTML = "";
  state.examples.forEach((ex) => {
    const b = document.createElement("button");
    b.type = "button";
    b.title = ex.title;
    const img = document.createElement("img");
    img.src = ex.image;
    img.alt = ex.title;
    img.loading = "lazy";
    b.appendChild(img);
    b.addEventListener("click", async () => {
      try {
        const res = await fetch(ex.image);
        if (!res.ok) throw new Error(`${ex.image}: HTTP ${res.status}`);
        const bytes = await res.arrayBuffer();
        selectGallery(b);
        setImage(bytes, res.headers.get("content-type") || "image/png", ex.title, ex);
      } catch (err) {
        fail(err, "Could not load the example");
      }
    });
    ui.gallery.appendChild(b);
  });
}

// -- classify -----------------------------------------------------------------

async function embedText(text) {
  const { query_prefix: prefix } = state.manifest;
  const out = await state.embedder(prefix + text, { pooling: "mean", normalize: true });
  return out.data;
}

let running = false;
let queued = false;

async function classify() {
  if (running) {
    queued = true;
    return;
  }
  running = true;
  ui.run.disabled = true;
  try {
    do {
      queued = false;
      await classifyOnce();
    } while (queued);
  } finally {
    running = false;
    ui.run.disabled = false;
  }
}

async function classifyOnce() {
  const text = normalizeQuery(ui.text.value || "");
  if (!text) {
    renderStatus("Write or generate a description first.");
    return;
  }
  state.error = null;
  try {
    const [index] = await Promise.all([loadIndex(), loadEmbedder()]);
    renderStatus("Embedding…");
    const t0 = performance.now();
    const query = await embedText(text);
    const t1 = performance.now();
    const params = {
      level: ui.level.value,
      topK: Number(ui.topK.value),
      excludeCodes: ui.noColours.checked ? ["29"] : [],
      principalOnly: ui.principal.checked,
    };
    const { matches, sims } = rank(index, query, params);
    const t2 = performance.now();
    renderGraph(index, matches, sims);
    renderResults(matches);
    const entry = state.manifest.index;
    ui.meta.textContent =
      `Vienna ${entry.edition} ${entry.lang} · ${entry.rows.toLocaleString()} entries · ` +
      `${state.manifest.web_model} ${state.manifest.web_dtype} · ` +
      `embed ${Math.round(t1 - t0)} ms · score ${Math.round(t2 - t1)} ms`;
    renderStatus();
    history.replaceState(null, "", `?${new URLSearchParams({ level: params.level, q: text })}`);
  } catch (err) {
    fail(err);
  }
}

function renderResults(matches) {
  ui.results.innerHTML = "";
  if (!matches.length) {
    ui.results.textContent = "No result.";
    return;
  }
  const table = document.createElement("table");
  table.innerHTML =
    "<thead><tr><th>#</th><th>Code</th><th>Level</th><th>Score</th><th>Sim.</th>" +
    "<th>Category &gt; division &gt; section</th></tr></thead>";
  const body = document.createElement("tbody");
  matches.forEach((m, i) => {
    const tr = document.createElement("tr");
    const cells = [
      String(i + 1),
      `${m.auxiliary ? "A " : ""}${m.pretty}`,
      m.level,
      m.score.toFixed(3),
      m.similarity.toFixed(3),
      m.text,
    ];
    cells.forEach((v, j) => {
      const td = document.createElement("td");
      td.textContent = v;
      if (j === 1) td.className = "code";
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  table.appendChild(body);
  ui.results.appendChild(table);
}

// -- graph: the results' paths merged into one tree ----------------------------

const G = { rowH: 34, gap: 96, padX: 14, padY: 16, nodeH: 24, charW: 7.4, badgeW: 34 };
const SVG = "http://www.w3.org/2000/svg";
const ROOT = -1; // virtual node above the categories, so category edges carry a value

function svgEl(tag, attrs = {}, text) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  return el;
}

/** Merge the result paths into one tree under a virtual root; assign a row to every node. */
function buildGraph(index, matches) {
  const nodes = new Map();
  nodes.set(ROOT, { i: ROOT, col: 0, parent: null, children: [], minRank: 1 });
  matches.forEach((m, r) => {
    const chain = chainOf(index, m.index);
    chain.forEach((i, d) => {
      if (!nodes.has(i)) {
        nodes.set(i, { i, col: d + 1, parent: d ? chain[d - 1] : ROOT, children: [], minRank: r + 1 });
      }
    });
    const leaf = nodes.get(m.index);
    leaf.match = m;
    leaf.rank = r + 1;
  });
  for (const n of nodes.values()) if (n.parent !== null) nodes.get(n.parent).children.push(n.i);
  const byRank = (a, b) => nodes.get(a).minRank - nodes.get(b).minRank || a - b;
  for (const n of nodes.values()) n.children.sort(byRank);
  let row = 0;
  const place = (n) => {
    if (!n.children.length) {
      n.row = row++;
      return;
    }
    n.children.forEach((c) => place(nodes.get(c)));
    const rows = n.children.map((c) => nodes.get(c).row);
    n.row = (Math.min(...rows) + Math.max(...rows)) / 2;
  };
  place(nodes.get(ROOT));
  return { nodes, rows: row };
}

function renderGraph(index, matches, sims) {
  ui.graph.innerHTML = "";
  hideTip();
  if (!matches.length) {
    ui.legend.hidden = true;
    return;
  }
  const { nodes, rows } = buildGraph(index, matches);
  for (const n of nodes.values()) {
    n.label = n.i === ROOT ? "Vienna" : `${index.auxiliary[n.i] ? "A " : ""}${index.code[n.i]}`;
    n.w = Math.max(28, 12 + n.label.length * G.charW);
  }
  const cols = Math.max(...[...nodes.values()].map((n) => n.col)) + 1;
  const colW = new Array(cols).fill(0);
  for (const n of nodes.values()) colW[n.col] = Math.max(colW[n.col], n.w);
  const colX = [];
  let x = G.padX;
  for (let c = 0; c < cols; c++) {
    colX.push(x);
    x += colW[c] + G.gap;
  }
  const width = x - G.gap + G.badgeW + G.padX;
  const height = rows * G.rowH + 2 * G.padY;
  const yOf = (row) => G.padY + row * G.rowH + G.rowH / 2;

  const svg = svgEl("svg", {
    width,
    height,
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Paths of the results through the Vienna Classification",
  });
  const edges = svgEl("g", { class: "edges" });
  const labels = svgEl("g", { class: "edge-labels" });
  const nodeLayer = svgEl("g", { class: "nodes" });
  svg.append(edges, labels, nodeLayer);

  for (const n of nodes.values()) {
    const nx = colX[n.col];
    const ny = yOf(n.row);
    const isResult = n.match !== undefined;
    if (n.parent !== null) {
      const p = nodes.get(n.parent);
      const x1 = colX[p.col] + p.w;
      const y1 = yOf(p.row);
      const xm = (x1 + nx) / 2;
      const d = `M${x1},${y1} C${xm},${y1} ${xm},${ny} ${nx},${ny}`;
      const value = isResult ? n.match.score : sims[n.i];
      const edge = svgEl("path", { d, class: `edge${isResult ? " result" : ""}` });
      const hit = svgEl("path", { d, class: "hit", tabindex: "0" });
      edges.append(edge, hit);
      const lx = (x1 + nx) / 2;
      const ly = (y1 + ny) / 2;
      const txt = value.toFixed(3);
      const lw = 10 + txt.length * 6.6;
      const pill = svgEl("g", { class: `edge-label${isResult ? " result" : ""}` });
      pill.append(
        svgEl("rect", { x: lx - lw / 2, y: ly - 9, width: lw, height: 18, rx: 9 }),
        svgEl("text", { x: lx, y: ly + 4, "text-anchor": "middle" }, txt),
      );
      labels.append(pill);
      const info = tipInfo(index, n, sims);
      hit.setAttribute("aria-label", info.aria);
      bindTip(hit, info, edge);
    }
    const g = svgEl("g", {
      class: `node${isResult ? " result" : ""}${n.i === ROOT ? " root" : ""}`,
      transform: `translate(${nx},${ny - G.nodeH / 2})`,
      tabindex: "0",
    });
    g.append(
      svgEl("rect", { width: n.w, height: G.nodeH, rx: 6 }),
      svgEl("text", { x: n.w / 2, y: G.nodeH / 2 + 4, "text-anchor": "middle" }, n.label),
    );
    if (isResult) {
      g.append(svgEl("text", { class: "badge", x: n.w + 8, y: G.nodeH / 2 + 4 }, `#${n.rank}`));
    }
    const info = tipInfo(index, n, sims);
    g.setAttribute("aria-label", info.aria);
    bindTip(g, info, g);
    nodeLayer.append(g);
  }
  ui.graph.appendChild(svg);
  ui.legend.hidden = false;
}

function tipInfo(index, n, sims) {
  const entry = state.manifest.index;
  if (n.i === ROOT) {
    const strong = `Vienna ${entry.edition} ${entry.lang}`;
    const title = "The whole classification. Each edge to a category carries that category's similarity.";
    return {
      lines: [{ strong, rest: ` · ${entry.rows.toLocaleString()} entries` }],
      code: "root",
      title,
      aria: `${strong}. ${title}`,
    };
  }
  const code = `${index.auxiliary[n.i] ? "A " : ""}${formatCode(index.code[n.i])}`;
  const level = `${LEVELS[index.level[n.i]]}${index.auxiliary[n.i] ? " (auxiliary)" : ""}`;
  const title = index.title[n.i] || "";
  const lines = [];
  if (n.match) {
    lines.push({ strong: `score ${n.match.score.toFixed(3)}`, rest: ` · result #${n.rank}` });
    lines.push({ rest: `similarity ${n.match.similarity.toFixed(3)}` });
  } else {
    lines.push({ strong: `similarity ${sims[n.i].toFixed(3)}`, rest: "" });
  }
  return {
    lines,
    code: `${code} · ${level}`,
    title,
    aria: `${code}, ${level}: ${title}. ${lines.map((l) => (l.strong || "") + l.rest).join(". ")}`,
  };
}

let lifted = null; // the element currently highlighted by the tooltip

function bindTip(target, info, lift) {
  const show = (e) => {
    if (lifted && lifted !== lift) lifted.classList.remove("hover");
    lifted = lift;
    lift.classList.add("hover");
    showTip(info, e);
  };
  const hide = () => {
    lift.classList.remove("hover");
    if (lifted === lift) lifted = null;
    hideTip();
  };
  target.addEventListener("pointerenter", show);
  // re-show on every move, so the tooltip always describes the element under the
  // pointer even when a browser skips an enter/leave pair between adjacent elements
  target.addEventListener("pointermove", (e) => {
    if (lifted !== lift || ui.tip.hidden) show(e);
    else positionTip(e.clientX, e.clientY);
  });
  target.addEventListener("pointerleave", hide);
  target.addEventListener("focus", (e) => {
    const r = e.target.getBoundingClientRect();
    lift.classList.add("hover");
    showTip(info, null);
    positionTip(r.left + r.width / 2, r.top + r.height / 2);
  });
  target.addEventListener("blur", hide);
}

function showTip(info, e) {
  ui.tip.innerHTML = "";
  for (const l of info.lines) {
    const p = document.createElement("div");
    if (l.strong) {
      const b = document.createElement("strong");
      b.textContent = l.strong;
      p.appendChild(b);
    }
    p.appendChild(document.createTextNode(l.rest));
    ui.tip.appendChild(p);
  }
  const code = document.createElement("div");
  code.className = "tip-code";
  code.textContent = info.code;
  const title = document.createElement("div");
  title.className = "tip-title";
  title.textContent = info.title;
  ui.tip.append(code, title);
  ui.tip.hidden = false;
  if (e) positionTip(e.clientX, e.clientY);
}

function positionTip(cx, cy) {
  const pad = 14;
  const w = ui.tip.offsetWidth;
  const h = ui.tip.offsetHeight;
  let left = cx + pad;
  let top = cy + pad;
  if (left + w > window.innerWidth - 8) left = cx - w - pad;
  if (top + h > window.innerHeight - 8) top = cy - h - pad;
  ui.tip.style.left = `${Math.max(8, left)}px`;
  ui.tip.style.top = `${Math.max(8, top)}px`;
}

function hideTip() {
  ui.tip.hidden = true;
  if (lifted) {
    lifted.classList.remove("hover");
    lifted = null;
  }
}

// -- setup --------------------------------------------------------------------

function fillSelect(select, values) {
  select.innerHTML = "";
  for (const v of values) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    select.appendChild(o);
  }
}

async function main() {
  try {
    const res = await fetch("manifest.json");
    if (!res.ok) throw new Error(`manifest.json: HTTP ${res.status}`);
    state.manifest = await res.json();
  } catch (err) {
    fail(err, "Could not load the index manifest");
    return;
  }
  fillSelect(ui.level, [AUTO_LEVEL, ...LEVELS]);
  ui.level.value = "section";
  if (!state.manifest.vision) ui.describe.hidden = true;
  await loadExamples();
  renderGallery();
  bindImageInput();

  const params = new URLSearchParams(location.search);
  if (params.get("level") && [AUTO_LEVEL, ...LEVELS].includes(params.get("level"))) {
    ui.level.value = params.get("level");
  }
  if (params.get("q")) ui.text.value = params.get("q");

  ui.run.addEventListener("click", classify);
  ui.describe.addEventListener("click", describeCurrent);
  window.addEventListener("scroll", hideTip, { passive: true });
  // the tooltip never outlives the pointer's stay in the graph
  ui.graph.addEventListener("pointerleave", hideTip);
  document.addEventListener("pointermove", (e) => {
    if (!ui.tip.hidden && !ui.graph.contains(e.target)) hideTip();
  });
  document.addEventListener("pointerdown", (e) => {
    if (!ui.graph.contains(e.target)) hideTip();
  });

  try {
    await Promise.all([loadIndex(), loadEmbedder()]);
    renderStatus(); // nothing runs until a button is clicked
  } catch (err) {
    fail(err, "Could not load the model or the index");
  }
}

main();
