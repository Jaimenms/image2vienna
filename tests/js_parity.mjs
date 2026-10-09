// Run the JavaScript scorer on an exported Space directory for a list of cases and
// print the matches as JSON. Driven by tests/test_web.py; needs Node >= 18.
//   node tests/js_parity.mjs <export_dir> <cases.json>
// cases.json: [{"query": [floats] | [[floats]...], "params": {...}} | {"split": text} | {"normalize": text}]
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [exportDir, casesPath] = process.argv.slice(2);
const { buildIndex, parseVectors, search, splitSentences, normalizeQuery } = await import(
  pathToFileURL(join(exportDir, "scorer.js")).href
);
const manifest = JSON.parse(readFileSync(join(exportDir, "manifest.json"), "utf8"));
const cases = JSON.parse(readFileSync(casesPath, "utf8"));

const entry = manifest.index;
const scheme = JSON.parse(readFileSync(join(exportDir, entry.scheme), "utf8"));
const raw = readFileSync(join(exportDir, entry.vectors));
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const vectors = parseVectors(buffer, { rows: entry.rows, dim: manifest.dim, encoding: entry.encoding });
const index = buildIndex(scheme, vectors);

const toQuery = (q) => (Array.isArray(q[0]) ? q.map((v) => Float32Array.from(v)) : Float32Array.from(q));
const out = cases.map((c) =>
  c.split !== undefined
    ? splitSentences(c.split)
    : c.normalize !== undefined
    ? normalizeQuery(c.normalize)
    : search(index, toQuery(c.query), c.params).map((m) => ({
        code: m.code,
        pretty: m.pretty,
        padded: m.padded,
        level: m.level,
        title: m.title,
        text: m.text,
        auxiliary: m.auxiliary,
        score: m.score,
        similarity: m.similarity,
        path_support: m.pathSupport,
        subtree_support: m.subtreeSupport,
      })),
);
process.stdout.write(JSON.stringify(out));
