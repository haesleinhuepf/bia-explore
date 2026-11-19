// Runs SAM 2.1 inside a Web Worker so the UI stays responsive.
import {
  Sam2Model,
  AutoProcessor,
  RawImage,
  Tensor,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';

const MODEL_ID = 'onnx-community/sam2.1-hiera-tiny-ONNX';

let model = null;
let processor = null;
let imageInputs = null;  // processor output for the current image (sizes, pixel values)
let embeddings = null;   // cached vision-encoder output for the current image

async function load() {
  const device = (await hasWebGPU()) ? 'webgpu' : 'wasm';
  const dtype = device === 'webgpu'
    ? { vision_encoder: 'fp16', prompt_encoder_mask_decoder: 'fp32' }
    : { vision_encoder: 'fp32', prompt_encoder_mask_decoder: 'fp32' };

  const progress_callback = (p) => {
    if (p.status === 'progress' && p.total) {
      self.postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
    }
  };

  [model, processor] = await Promise.all([
    Sam2Model.from_pretrained(MODEL_ID, { device, dtype, progress_callback }),
    AutoProcessor.from_pretrained(MODEL_ID),
  ]);
  return { device };
}

async function hasWebGPU() {
  if (!navigator.gpu) return false;
  try {
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function embed({ data, width, height }) {
  embeddings = null;
  const image = new RawImage(new Uint8ClampedArray(data), width, height, 4).rgb();
  imageInputs = await processor(image);
  embeddings = await model.get_image_embeddings(imageInputs);
  return { width, height };
}

// objects: [{ box: [x1,y1,x2,y2] | null, points: [{x, y, label}] }] in image pixel coordinates
async function segment({ objects }) {
  if (!embeddings) throw new Error('Image has not been embedded yet.');
  const { original_sizes, reshaped_input_sizes } = imageInputs;
  const [height, width] = original_sizes[0];

  const results = [];
  for (const obj of objects) {
    const inputs = {};
    if (obj.points.length) {
      inputs.input_points = processor.reshape_input_points(
        [obj.points.map((p) => [p.x, p.y])], original_sizes, reshaped_input_sizes,
      );
      inputs.input_labels = new Tensor(
        'int64', BigInt64Array.from(obj.points.map((p) => BigInt(p.label))), [1, 1, obj.points.length],
      );
    }
    if (obj.box) {
      inputs.input_boxes = processor.reshape_input_points(
        [[obj.box]], original_sizes, reshaped_input_sizes, true,
      );
    }

    const outputs = await model({ ...embeddings, ...inputs });
    const [masks] = await processor.post_process_masks(
      outputs.pred_masks, original_sizes, reshaped_input_sizes,
    );

    // SAM proposes 3 candidate masks; keep the one with the highest predicted IoU.
    const scores = Array.from(outputs.iou_scores.data);
    const best = scores.indexOf(Math.max(...scores));
    const size = width * height;
    const mask = masks.data.slice(best * size, (best + 1) * size);
    results.push({ mask, score: scores[best] });
  }
  return { width, height, results };
}

const handlers = { load, embed, segment };

self.onmessage = async ({ data }) => {
  const { id, type, payload } = data;
  try {
    const result = await handlers[type](payload);
    const transfer = type === 'segment' ? result.results.map((r) => r.mask.buffer) : [];
    self.postMessage({ id, result }, transfer);
  } catch (err) {
    console.error(err);
    self.postMessage({ id, error: String(err?.message ?? err) });
  }
};
