// Web Worker that describes an image with a small vision-language model through
// transformers.js, so the page keeps painting while the model loads and generates.
// Messages in: {id, type: "describe", modelId, dtype, device, f16, bytes, mime, prompt, maxNewTokens}
// Messages out: {type: "progress", file, loaded, total} while loading, {type: "fallback", device,
// message} when an attempt fails, {type: "ready", device, dtype}, {type: "generating", tokens} per
// token, then {id, type: "result", text, device} or {id, type: "error", message}.
import {
  env,
  AutoProcessor,
  AutoModelForVision2Seq,
  RawImage,
  TextStreamer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

env.allowLocalModels = false;

let loaded = null; // {modelId, processor, model, device, dtype}
let loading = null;

// Weights per device: fp16 needs the WebGPU "shader-f16" feature (the main thread
// checks it); without it fp32 runs on WebGPU, and WASM takes q8/fp32. Each attempt
// that fails to load or to run falls through to the next.
const FP32 = { embed_tokens: "fp32", vision_encoder: "fp32", decoder_model_merged: "q4" };
const WASM = { embed_tokens: "fp32", vision_encoder: "q8", decoder_model_merged: "q4" };

function attemptsFor(dtype, device, f16) {
  const out = [];
  if (device === "webgpu") {
    if (f16) out.push({ device: "webgpu", dtype });
    out.push({ device: "webgpu", dtype: FP32 });
  }
  out.push({ device: "wasm", dtype: WASM });
  return out;
}

async function loadOne(modelId, attempt) {
  const progress_callback = (e) => {
    if (e.status === "progress" && e.file) {
      postMessage({ type: "progress", file: e.file, loaded: e.loaded, total: e.total });
    }
  };
  const processor = await AutoProcessor.from_pretrained(modelId, { progress_callback });
  const model = await AutoModelForVision2Seq.from_pretrained(modelId, {
    ...attempt,
    progress_callback,
  });
  return { modelId, processor, model, ...attempt };
}

function load(modelId, dtype, device, f16) {
  if (loaded && loaded.modelId === modelId) return Promise.resolve(loaded);
  if (!loading) {
    loading = (async () => {
      let lastError = null;
      for (const attempt of attemptsFor(dtype, device, f16)) {
        try {
          loaded = await loadOne(modelId, attempt);
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
  const messages = [
    { role: "user", content: [{ type: "image" }, { type: "text", text: m.prompt }] },
  ];
  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  const inputs = await processor(text, [image], { do_image_splitting: false });
  let tokens = 0;
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
  return decoded.trim();
}

/** Describe with the loaded model; a failure at run time on WebGPU retries on WASM. */
async function describe(m) {
  const entry = await load(m.modelId, m.dtype, m.device, m.f16);
  try {
    return { text: await generate(entry, m), device: entry.device };
  } catch (err) {
    if (entry.device === "wasm") throw err;
    postMessage({ type: "fallback", device: "wasm", message: err.message || String(err) });
    loaded = null;
    loading = null;
    const retry = await load(m.modelId, m.dtype, "wasm", false);
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
