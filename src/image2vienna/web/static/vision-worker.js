// Web Worker that describes an image with a small vision-language model through
// transformers.js, so the page keeps painting while the model loads and generates.
// Two kinds of model: "florence" (Florence-2, a captioner driven by a task token such as
// <MORE_DETAILED_CAPTION>; the light model) and "chat" (SmolVLM and the like, an
// instruction through the chat template).
// Messages in: {id, type: "describe", modelId, kind, dtype, device, f16, bytes, mime, prompt, maxNewTokens}
// Messages out: {type: "progress", file, loaded, total} while loading, {type: "fallback", device,
// message} when an attempt fails, {type: "ready", device, dtype}, {type: "generating", tokens} per
// token, then {id, type: "result", text, device} or {id, type: "error", message}.
import {
  env,
  AutoProcessor,
  AutoModelForVision2Seq,
  Florence2ForConditionalGeneration,
  RawImage,
  TextStreamer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

env.allowLocalModels = false;

let loaded = null; // {modelId, processor, model, device, dtype}
let loading = null;

// Weights per device: fp16 needs the WebGPU "shader-f16" feature (the main thread
// checks it); without it fp32 runs on WebGPU, and WASM takes q8/fp32. Each attempt
// that fails to load or to run falls through to the next.
const FALLBACKS = {
  florence: {
    webgpu: { embed_tokens: "fp32", vision_encoder: "fp32", encoder_model: "q4", decoder_model_merged: "q4" },
    wasm: { embed_tokens: "fp32", vision_encoder: "q8", encoder_model: "q8", decoder_model_merged: "q8" },
  },
  chat: {
    webgpu: { embed_tokens: "fp32", vision_encoder: "fp32", decoder_model_merged: "q4" },
    wasm: { embed_tokens: "fp32", vision_encoder: "q8", decoder_model_merged: "q4" },
  },
};

function attemptsFor(kind, dtype, device, f16) {
  const fb = FALLBACKS[kind] || FALLBACKS.chat;
  const out = [];
  if (device === "webgpu") {
    if (f16) out.push({ device: "webgpu", dtype });
    out.push({ device: "webgpu", dtype: fb.webgpu });
  }
  out.push({ device: "wasm", dtype: fb.wasm });
  return out;
}

async function loadOne(modelId, kind, attempt) {
  const progress_callback = (e) => {
    if (e.status === "progress" && e.file) {
      postMessage({ type: "progress", file: e.file, loaded: e.loaded, total: e.total });
    }
  };
  const processor = await AutoProcessor.from_pretrained(modelId, { progress_callback });
  const Model = kind === "florence" ? Florence2ForConditionalGeneration : AutoModelForVision2Seq;
  const model = await Model.from_pretrained(modelId, { ...attempt, progress_callback });
  return { modelId, kind, processor, model, ...attempt };
}

function load(modelId, kind, dtype, device, f16) {
  if (loaded && loaded.modelId === modelId) return Promise.resolve(loaded);
  if (!loading) {
    loading = (async () => {
      let lastError = null;
      for (const attempt of attemptsFor(kind, dtype, device, f16)) {
        try {
          loaded = await loadOne(modelId, kind, attempt);
          postMessage({ type: "ready", device: loaded.device, dtype: loaded.dtype });
          return loaded;
        } catch (err) {
          lastError = err;
          postMessage({ type: "fallback", device: attempt.device, message: err.message || String(err) });
        }
      }
      throw lastError;
    })().catch((err) => {
      loading = null; // let a later request retry
      throw err;
    });
  }
  return loading;
}

async function generate(entry, m) {
  const { processor, model } = entry;
  const image = await RawImage.fromBlob(new Blob([m.bytes], { type: m.mime || "image/png" }));
  let tokens = 0;
  if (entry.kind === "florence") {
    const task = m.prompt && m.prompt.startsWith("<") ? m.prompt : "<MORE_DETAILED_CAPTION>";
    const visionInputs = await processor(image);
    const textInputs = processor.tokenizer(processor.construct_prompts(task));
    const streamer = new TextStreamer(processor.tokenizer, {
      skip_prompt: false,
      skip_special_tokens: true,
      callback_function: () => postMessage({ type: "generating", tokens: ++tokens }),
    });
    const out = await model.generate({
      ...textInputs,
      ...visionInputs,
      max_new_tokens: m.maxNewTokens || 120,
      do_sample: false,
      streamer,
    });
    const [decoded] = processor.batch_decode(out, { skip_special_tokens: false });
    const result = processor.post_process_generation(decoded, task, image.size);
    return cleanDescription(String(result[task] ?? decoded));
  }
  const messages = [
    { role: "user", content: [{ type: "image" }, { type: "text", text: m.prompt }] },
  ];
  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  const inputs = await processor(text, [image], { do_image_splitting: false });
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: () => postMessage({ type: "generating", tokens: ++tokens }),
  });
  const out = await model.generate({
    ...inputs,
    max_new_tokens: m.maxNewTokens || 220,
    do_sample: false,
    streamer,
  });
  const generated = out.slice(null, [inputs.input_ids.dims.at(-1), null]);
  const [decoded] = processor.batch_decode(generated, { skip_special_tokens: true });
  return cleanDescription(decoded);
}

const SENTENCE_END_RE = /(?<=[.!?])\s+(?=[A-Z0-9"'(])/;

/**
 * Tidy the model's output as image2vienna.textnorm.clean_description does: drop
 * repeated sentences (small models loop) and fragments without a word, keeping the
 * first occurrences in order.
 */
export function cleanDescription(text) {
  const joined = text.split(/\s+/).filter(Boolean).join(" ");
  const parts = joined.split(SENTENCE_END_RE).map((p) => p.trim()).filter(Boolean);
  const seen = new Set();
  const kept = [];
  for (const raw of parts.length ? parts : [text.trim()]) {
    const sentence = raw.replace(/[\s,;:]+[^A-Za-z]*$/, "").trim();
    const key = sentence.toLowerCase().split(/\s+/).join(" ");
    if (!sentence || seen.has(key) || !/[A-Za-z]{3,}/.test(sentence)) continue;
    seen.add(key);
    kept.push(sentence);
  }
  return kept.join(" ").trim();
}

/** Describe with the loaded model; a failure at run time on WebGPU retries on WASM. */
async function describe(m) {
  const entry = await load(m.modelId, m.kind, m.dtype, m.device, m.f16);
  try {
    return { text: await generate(entry, m), device: entry.device };
  } catch (err) {
    if (entry.device === "wasm") throw err;
    postMessage({ type: "fallback", device: "wasm", message: err.message || String(err) });
    loaded = null;
    loading = null;
    const retry = await load(m.modelId, m.kind, m.dtype, "wasm", false);
    return { text: await generate(retry, m), device: retry.device };
  }
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type !== "describe") return;
  try {
    const { text, device } = await describe(m);
    postMessage({ id: m.id, type: "result", text, device });
  } catch (err) {
    postMessage({ id: m.id, type: "error", message: err.message || String(err) });
  }
};
