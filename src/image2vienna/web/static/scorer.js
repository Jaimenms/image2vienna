// JavaScript port of image2vienna/search/scorer.py: hierarchical scoring on top of
// flat cosine similarity over the Vienna Classification (category > division >
// section). Pure functions, no DOM, no fetch: the browser page (app.js) and the Node
// parity test (tests/js_parity.mjs) both import it.
//
// Keep in step with the Python module. Every heuristic here mirrors a function
// there by name: pathSupport, subtreeSupport, beamDescend, autoDescend,
// distinctBranches, mask. tests/test_web.py runs both on the mini scheme and compares.

export const LEVELS = ["category", "division", "section"];
export const AUTO_LEVEL = "auto";
export const PATH_SEPARATOR = " > ";

export const DEFAULT_WEIGHTS = { own: 1.0, path: 0.0, subtree: 0.0 }; // Weights in scorer.py
export const DEFAULT_BEAM = { category: 8, division: 20 };
export const DEFAULT_PARAMS = {
  level: "section",
  topK: 10,
  gap: null, // drop results scoring more than this below the best one
  autoMargin: 0.02, // auto level: descend while best child >= parent - margin
  autoRoots: null, // auto level: divisions to start from; null = max(5, topK)
  dedupeBranches: true, // drop ancestors/descendants of a higher-ranked result
  principalOnly: false, // leave auxiliary (A) sections out
  excludeCodes: [], // codes whose subtree is left out, e.g. ["29"]
};

// -- index ------------------------------------------------------------------

/**
 * Build the in-memory index from the exported columns and vectors.
 * scheme: {code: string[], level: int[], parent: int[], title: string[], auxiliary: int[]}
 * vectors: {encoding: "int8"|"float32", rows, dim, data: Int8Array|Float32Array, scales?: Float32Array}
 */
export function buildIndex(scheme, vectors) {
  const n = scheme.code.length;
  if (vectors.rows !== n) {
    throw new Error(`vectors have ${vectors.rows} rows, scheme has ${n}`);
  }
  const parent = Int32Array.from(scheme.parent);
  const children = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) {
    const p = parent[i];
    if (p >= i) throw new Error("Index rows must be in depth-first order (parents before children)");
    if (p >= 0) children[p].push(i);
  }
  return {
    rows: n,
    dim: vectors.dim,
    code: scheme.code,
    level: Int8Array.from(scheme.level),
    parent,
    children,
    title: scheme.title,
    auxiliary: Uint8Array.from(scheme.auxiliary.map((a) => (a ? 1 : 0))),
    vectors,
  };
}

/** Parse vectors.bin as written by image2vienna.web.export.write_vectors. */
export function parseVectors(buffer, meta) {
  const { rows, dim, encoding } = meta;
  if (encoding === "int8") {
    const scales = new Float32Array(buffer, 0, rows);
    const data = new Int8Array(buffer, rows * 4, rows * dim);
    return { encoding, rows, dim, data, scales };
  }
  if (encoding === "float32") {
    return { encoding, rows, dim, data: new Float32Array(buffer, 0, rows * dim) };
  }
  throw new Error(`unknown vector encoding ${encoding}`);
}

/** Canonical code (1.1.2) as is, or padded EUIPO style (01.01.02). */
export function formatCode(code, padded = false) {
  if (!padded) return code;
  return code.split(".").map((p) => p.padStart(2, "0")).join(".");
}

/** Row chain from the category down to row i. */
export function chainOf(index, i) {
  const chain = [];
  for (let node = i; node >= 0; node = index.parent[node]) chain.push(node);
  return chain.reverse();
}

/** The text an entry was embedded with: titles top-down (titles-only index). */
export function textAt(index, i) {
  return chainOf(index, i)
    .map((s) => index.title[s])
    .filter((t) => t)
    .join(PATH_SEPARATOR);
}

// -- query normalisation (image2vienna/textnorm.py) ------------------------------

export function collapseWhitespace(text) {
  return text
    .trim()
    .split(/\r?\n/)
    .map((line) => line.split(/[ \t\r\f\v]+/).filter(Boolean).join(" "))
    .join("\n");
}

export function lowercaseIfShouting(text, threshold = 0.7) {
  const letters = Array.from(text).filter((c) => /\p{L}/u.test(c));
  if (!letters.length) return text;
  const upper = letters.filter((c) => c !== c.toLowerCase() && c === c.toUpperCase()).length;
  return upper / letters.length >= threshold ? text.toLowerCase() : text;
}

export function normalizeQuery(text) {
  return lowercaseIfShouting(collapseWhitespace(text)).split(/\s+/).filter(Boolean).join(" ");
}

const SENTENCE_END_RE = /(?<=[.!?])\s+(?=[A-Z0-9"'(])/;

/** Sentences of a description, for per-element scoring; never empty. */
export function splitSentences(text) {
  const joined = text.split(/\s+/).filter(Boolean).join(" ");
  const parts = joined.split(SENTENCE_END_RE).map((p) => p.trim()).filter(Boolean);
  return parts.length ? parts : [text.trim()];
}

/** Unit-length mean of several unit vectors (the "mean" chunking policy). */
export function meanVector(vectors) {
  const dim = vectors[0].length;
  const out = new Float32Array(dim);
  for (const v of vectors) for (let j = 0; j < dim; j++) out[j] += v[j];
  let norm = 0;
  for (let j = 0; j < dim; j++) norm += out[j] * out[j];
  norm = Math.sqrt(norm) || 1;
  for (let j = 0; j < dim; j++) out[j] /= norm;
  return out;
}

// -- scoring ------------------------------------------------------------------

function similarities(index, query) {
  const { rows, dim, data, scales, encoding } = index.vectors;
  // an array of vectors comes from sentence chunking: an entry scores by its best one
  const queries = Array.isArray(query) ? query : [query];
  const sims = new Float64Array(rows).fill(-Infinity);
  for (const q of queries) {
    if (q.length !== dim) throw new Error(`query has ${q.length} dimensions, index has ${dim}`);
    for (let i = 0; i < rows; i++) {
      const off = i * dim;
      let s = 0;
      for (let j = 0; j < dim; j++) s += data[off + j] * q[j];
      s = encoding === "int8" ? s * scales[i] : s;
      if (s > sims[i]) sims[i] = s;
    }
  }
  return sims;
}

/** Rows to leave out: auxiliary sections and/or the subtrees of excludeCodes. */
function mask(index, p) {
  if (!p.principalOnly && !(p.excludeCodes && p.excludeCodes.length)) return null;
  const out = new Uint8Array(index.rows);
  if (p.principalOnly) for (let i = 0; i < index.rows; i++) if (index.auxiliary[i]) out[i] = 1;
  for (const code of p.excludeCodes || []) {
    const prefix = `${code}.`;
    for (let i = 0; i < index.rows; i++) {
      const c = index.code[i];
      if (c === code || c.startsWith(prefix)) out[i] = 1;
    }
  }
  return out;
}

/** Mean similarity of the ancestors; equals own similarity for roots. */
function pathSupport(index, sims) {
  const n = index.rows;
  const total = new Float64Array(n);
  const count = new Int32Array(n);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = index.parent[i]; // parents precede children
    if (p >= 0) {
      total[i] = total[p] + sims[p];
      count[i] = count[p] + 1;
    }
    out[i] = count[i] > 0 ? total[i] / count[i] : sims[i];
  }
  return out;
}

/** Max value over the node and everything below it. */
function subtreeSupport(index, values) {
  const best = Float64Array.from(values);
  for (let i = index.rows - 1; i >= 0; i--) {
    const p = index.parent[i]; // children precede parents in reverse
    if (p >= 0 && best[i] > best[p]) best[p] = best[i];
  }
  return best;
}

/** Stable descending sort of row indices by a value array (Python's sorted(key=-v)). */
function sortDesc(items, values) {
  return items.slice().sort((a, b) => values[b] - values[a]);
}

/** Walk down from the categories, keeping the parents whose subtree scores best. */
function beamDescend(index, bestBelow, target, beam) {
  const targetRank = LEVELS.indexOf(target);
  let frontier = [];
  for (let i = 0; i < index.rows; i++) if (index.level[i] === 0) frontier.push(i);
  for (let rank = 0; rank < LEVELS.length; rank++) {
    if (rank === targetRank) return frontier;
    const level = LEVELS[rank];
    frontier = sortDesc(frontier, bestBelow).slice(0, beam[level]);
    frontier = frontier.flatMap((i) => index.children[i]);
  }
  return frontier;
}

function argmaxFirst(items, values) {
  let best = items[0];
  for (const i of items) if (values[i] > values[best]) best = i;
  return best;
}

/**
 * From the best divisions, walk down while the best child's subtree promises
 * something within autoMargin of the current node, then answer with the node on
 * the walked path whose own similarity is highest (deepest on ties).
 */
function autoDescend(index, sims, score, bestBelow, p) {
  const nRoots = p.autoRoots ?? Math.max(5, p.topK);
  let roots = beamDescend(index, bestBelow, "division", p.beam);
  roots = sortDesc(roots, bestBelow).slice(0, nRoots);
  const out = [];
  for (const i of roots) {
    const path = [i];
    let node = i;
    while (index.children[node].length) {
      const bestChild = argmaxFirst(index.children[node], bestBelow);
      if (bestBelow[bestChild] < score[node] - p.autoMargin) break;
      node = bestChild;
      path.push(node);
    }
    const best = argmaxFirst(path.slice().reverse(), sims);
    if (!out.includes(best)) out.push(best);
  }
  return sortDesc(out, score).slice(0, p.topK);
}

function isAncestorOrSelf(index, a, i) {
  for (let node = i; node >= 0; node = index.parent[node]) if (node === a) return true;
  return false;
}

/** Keep a result only if no accepted result sits on its path or below it. */
function distinctBranches(index, ranked) {
  const accepted = [];
  for (const i of ranked) {
    const sameBranch = accepted.some(
      (a) => isAncestorOrSelf(index, a, i) || isAncestorOrSelf(index, i, a),
    );
    if (!sameBranch) accepted.push(i);
  }
  return accepted;
}

/**
 * Rank Vienna entries for a unit query vector (or an array of them, scored by the
 * best one). Returns matches ordered by score with the same fields as
 * image2vienna.search.Match plus `pretty`, `padded` and `index` (the row).
 */
export function search(index, query, params = {}) {
  return rank(index, query, params).matches;
}

/**
 * Like search, but also returns the per-row arrays the matches were computed from
 * (`sims`, `scores`, `pathSupport`, `subtreeSupport`), so a caller can show how the
 * ancestors of a result scored. The Python package has no counterpart; `search`
 * is the function the parity test covers.
 */
export function rank(index, query, params = {}) {
  const p = {
    ...DEFAULT_PARAMS,
    ...params,
    weights: { ...DEFAULT_WEIGHTS, ...(params.weights || {}) },
    beam: { ...DEFAULT_BEAM, ...(params.beam || {}) },
  };
  if (p.level !== AUTO_LEVEL && !LEVELS.includes(p.level)) {
    throw new Error(`level must be one of ${LEVELS.join(", ")} or ${AUTO_LEVEL}`);
  }
  let sims = similarities(index, query);
  const masked = mask(index, p);
  if (masked) {
    sims = sims.map((s, i) => (masked[i] ? -1.0 : s));
  }
  const path = pathSupport(index, sims);
  const subtree = subtreeSupport(index, sims);
  const w = p.weights;
  const score = new Float64Array(index.rows);
  for (let i = 0; i < index.rows; i++) {
    score[i] = w.own * sims[i] + w.path * path[i] + w.subtree * subtree[i];
  }
  const bestBelow = subtreeSupport(index, score);

  let chosen;
  if (p.level === AUTO_LEVEL) {
    chosen = autoDescend(index, sims, score, bestBelow, p);
  } else {
    chosen = beamDescend(index, bestBelow, p.level, p.beam);
    chosen = sortDesc(chosen, score).slice(0, p.topK);
  }
  if (masked) chosen = chosen.filter((i) => !masked[i]);
  if (p.gap !== null && p.gap !== undefined && chosen.length) {
    const best = score[chosen[0]];
    chosen = chosen.filter((i) => score[i] >= best - p.gap);
  }
  if (p.dedupeBranches) chosen = distinctBranches(index, chosen);

  const matches = chosen.map((i) => ({
    index: i,
    code: index.code[i],
    pretty: formatCode(index.code[i]),
    padded: formatCode(index.code[i], true),
    level: LEVELS[index.level[i]],
    title: index.title[i],
    text: textAt(index, i),
    score: score[i],
    similarity: sims[i],
    pathSupport: path[i],
    subtreeSupport: subtree[i],
    auxiliary: Boolean(index.auxiliary[i]),
  }));
  return { matches, sims, scores: score, pathSupport: path, subtreeSupport: subtree };
}
