const MAX_SIDE = 1600; // captured images are downscaled to this; SAM works at 1024 anyway
const COLORS = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
  '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
  '#dcbeff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
];

const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

// ---------- Worker RPC ----------

const worker = new Worker('worker.js', { type: 'module' });
const pending = new Map();
let nextId = 0;
const downloads = {};

worker.onmessage = ({ data }) => {
  if (data.type === 'progress') {
    downloads[data.file] = data;
    const loaded = Object.values(downloads).reduce((s, d) => s + d.loaded, 0);
    const total = Object.values(downloads).reduce((s, d) => s + d.total, 0);
    $('model-status').textContent = `Downloading model… ${Math.round((100 * loaded) / total)}%`;
    return;
  }
  const { resolve, reject } = pending.get(data.id);
  pending.delete(data.id);
  data.error ? reject(new Error(data.error)) : resolve(data.result);
};

function rpc(type, payload, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, payload }, transfer);
  });
}

const modelReady = rpc('load').then(
  ({ device }) => {
    $('model-status').textContent = `SAM 2.1 ready (${device === 'webgpu' ? 'WebGPU' : 'CPU / WASM'})`;
  },
  (err) => {
    $('model-status').textContent = 'Model failed to load';
    toast(`Could not load model: ${err.message}`);
    throw err;
  },
);
modelReady.catch(() => {});

// ---------- State ----------

const image = document.createElement('canvas'); // the captured picture
const overlay = document.createElement('canvas'); // rendered masks
let embedding = null; // promise resolving once the current image is embedded
let embeddingQueue = Promise.resolve();
let embeddingReady = false;
let embeddingFailed = false;
let hasImage = false;
let busy = false;
let objects = [];     // { id, color, box, stroke, points: [{x, y, label}], mask, score, dirty }
let selectedId = null;
let tool = 'scribble';
let drag = null;      // in-progress box or scribble
let history = [];     // undo stack
let objectCounter = 0;
let lastClearTime = 0;

// ---------- Camera / image acquisition ----------

let stream = null;
let facingMode = 'environment';
let cameraLoading = false;

async function startCamera() {
  stopCamera();
  if (!navigator.mediaDevices?.getUserMedia) {
    toast('Camera needs HTTPS (or localhost). Use "Upload picture" instead.');
    return;
  }
  cameraLoading = true;
  updateUI();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    if (hasImage) newPicture();
    $('video-placeholder').hidden = true;
  } catch (err) {
    stopCamera();
    $('video-placeholder').hidden = false;
    toast(`Camera unavailable: ${err.message}`);
  } finally {
    cameraLoading = false;
    updateUI();
  }
}

function stopCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
}

function capture() {
  setImage(video, video.videoWidth, video.videoHeight);
  stopCamera();
  updateUI();
}

async function upload(file) {
  if (!file) return;
  try {
    const bitmap = await createImageBitmap(file);
    setImage(bitmap, bitmap.width, bitmap.height);
    bitmap.close();
    stopCamera();
    updateUI();
  } catch (err) {
    toast(`Could not open image: ${err.message}`);
  }
}

function setImage(source, w, h) {
  const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
  image.width = overlay.width = canvas.width = Math.round(w * scale);
  image.height = overlay.height = canvas.height = Math.round(h * scale);
  image.getContext('2d').drawImage(source, 0, 0, image.width, image.height);

  objects = [];
  history = [];
  selectedId = null;
  objectCounter = 0;
  lastClearTime = 0;
  hasImage = true;
  embeddingReady = false;
  embeddingFailed = false;
  rebuildOverlay();

  $('acquire').hidden = true;
  $('annotate').hidden = false;
  render();
  updateUI();

  // Start the expensive image-encoder pass right away, so segmentation is quick later.
  const { data, width, height } = image.getContext('2d').getImageData(0, 0, image.width, image.height);
  embedding = embeddingQueue.catch(() => {}).then(() => modelReady)
    .then(() => rpc('embed', { data: data.buffer, width, height }, [data.buffer]));
  embeddingQueue = embedding;
  const currentEmbedding = embedding;
  embedding.then(() => {
    if (embedding === currentEmbedding && hasImage) {
      embeddingReady = true;
      updateUI();
    }
  }, (err) => {
    if (embedding === currentEmbedding && hasImage) {
      embeddingFailed = true;
      updateUI();
      toast(`Embedding failed: ${err.message}`);
    }
  });
}

function newPicture() {
  hasImage = false;
  embedding = null;
  embeddingReady = false;
  objects = [];
  history = [];
  selectedId = null;
  drag = null;
  lastClearTime = 0;
  $('annotate').hidden = true;
  $('acquire').hidden = false;
  $('video-placeholder').hidden = false;
  $('file-input').value = '';
  updateUI();
}

// ---------- Annotation ----------

function canvasPoint(e) {
  const r = canvas.getBoundingClientRect();
  return {
    x: Math.min(canvas.width, Math.max(0, ((e.clientX - r.left) * canvas.width) / r.width)),
    y: Math.min(canvas.height, Math.max(0, ((e.clientY - r.top) * canvas.height) / r.height)),
  };
}

function addObject(props) {
  const obj = {
    id: ++objectCounter,
    color: COLORS[(objectCounter - 1) % COLORS.length],
    box: null,
    stroke: null,
    points: [],
    mask: null,
    score: null,
    dirty: true,
    ...props,
  };
  objects.push(obj);
  selectedId = obj.id;
  history.push({ type: 'add-object', id: obj.id });
  return obj;
}

function addPoint(p, label) {
  let obj = objects.find((o) => o.id === selectedId);
  if (!obj) {
    if (label === 0) {
      toast('Select an object first, then add exclude points to it.');
      return;
    }
    addObject({ points: [{ ...p, label }] });
    return;
  }
  obj.points.push({ ...p, label });
  obj.dirty = true;
  history.push({ type: 'add-point', id: obj.id });
}

// Turn a freehand stroke into a handful of positive points spread along its length.
function samplePoints(stroke) {
  const seg = [0];
  for (let i = 1; i < stroke.length; i++) {
    seg.push(seg[i - 1] + Math.hypot(stroke[i].x - stroke[i - 1].x, stroke[i].y - stroke[i - 1].y));
  }
  const length = seg[seg.length - 1];
  const diag = Math.hypot(canvas.width, canvas.height);
  const n = Math.max(1, Math.min(8, Math.round(length / (diag * 0.04))));
  const points = [];
  for (let k = 0; k < n; k++) {
    const target = ((k + 0.5) / n) * length;
    const i = Math.max(0, seg.findIndex((s) => s >= target));
    points.push({ x: stroke[i].x, y: stroke[i].y, label: 1, implicit: true });
  }
  return points;
}

canvas.addEventListener('pointerdown', (e) => {
  if (busy) return;
  canvas.setPointerCapture(e.pointerId);
  const p = canvasPoint(e);
  if (tool === 'box') drag = { type: 'box', start: p, end: p };
  else if (tool === 'scribble') drag = { type: 'scribble', stroke: [p] };
  else {
    addPoint(p, tool === 'pos' ? 1 : 0);
    afterEdit();
  }
});

canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const p = canvasPoint(e);
  if (drag.type === 'box') drag.end = p;
  else drag.stroke.push(p);
  render();
});

const endDrag = () => {
  if (!drag) return;
  const minSize = Math.max(canvas.width, canvas.height) * 0.01;
  if (drag.type === 'box') {
    const { start: a, end: b } = drag;
    if (Math.abs(a.x - b.x) > minSize && Math.abs(a.y - b.y) > minSize) {
      addObject({ box: [Math.min(a.x, b.x), Math.min(a.y, b.y), Math.max(a.x, b.x), Math.max(a.y, b.y)] });
    }
  } else if (drag.stroke.length > 1) {
    addObject({ stroke: drag.stroke, points: samplePoints(drag.stroke) });
  } else {
    // A tap in scribble mode acts as a single positive point on a new object.
    addObject({ points: [{ ...drag.stroke[0], label: 1 }] });
  }
  drag = null;
  afterEdit();
};
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', () => { drag = null; render(); });

function undo() {
  const step = history.pop();
  if (!step) return;
  if (step.type === 'add-object') {
    removeObject(step.id, false);
  } else {
    const obj = objects.find((o) => o.id === step.id);
    obj.points.pop();
    obj.dirty = true;
    selectedId = obj.id;
  }
  afterEdit();
}

function removeObject(id, record = true) {
  objects = objects.filter((o) => o.id !== id);
  if (record) history = history.filter((h) => h.id !== id);
  if (selectedId === id) selectedId = objects.at(-1)?.id ?? null;
  rebuildOverlay();
}

function afterEdit() {
  render();
  updateUI();
}

// ---------- Segmentation ----------

async function segmentAll() {
  const todo = objects.filter((o) => o.dirty);
  if (!todo.length) return;
  setBusy('Segmenting…');
  try {
    await embedding;
    const { results } = await rpc('segment', {
      objects: todo.map((o) => ({ box: o.box, points: o.points })),
    });
    todo.forEach((o, i) => {
      o.mask = results[i].mask;
      o.score = results[i].score;
      o.dirty = false;
    });
    rebuildOverlay();
  } catch (err) {
    toast(`Segmentation failed: ${err.message}`);
  } finally {
    setBusy(null);
    afterEdit();
  }
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// Paint every mask as a translucent fill with an opaque outline.
function rebuildOverlay() {
  const { width: w, height: h } = overlay;
  const octx = overlay.getContext('2d');
  octx.clearRect(0, 0, w, h);
  if (!objects.some((o) => o.mask)) return;
  const out = octx.createImageData(w, h);
  const px = out.data;
  for (const o of objects) {
    if (!o.mask) continue;
    const [r, g, b] = hexToRgb(o.color);
    const m = o.mask;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!m[i]) continue;
        const edge = x === 0 || y === 0 || x === w - 1 || y === h - 1
          || !m[i - 1] || !m[i + 1] || !m[i - w] || !m[i + w];
        px[4 * i] = r;
        px[4 * i + 1] = g;
        px[4 * i + 2] = b;
        px[4 * i + 3] = edge ? 255 : 120;
      }
    }
  }
  octx.putImageData(out, 0, 0);
}

// ---------- Rendering ----------

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0);
  ctx.drawImage(overlay, 0, 0);
  for (const o of objects) drawAnnotation(o, o.id === selectedId);
  drawDrag();
}

function lineWidth() {
  return Math.max(2, Math.max(canvas.width, canvas.height) / 400);
}

function drawAnnotation(o, selected) {
  const lw = lineWidth() * (selected ? 1.6 : 1);
  ctx.strokeStyle = o.color;
  ctx.lineWidth = lw;
  ctx.lineCap = ctx.lineJoin = 'round';
  ctx.setLineDash(o.dirty && o.mask ? [lw * 3, lw * 2] : []); // dashed = mask is outdated
  if (o.box) {
    const [x1, y1, x2, y2] = o.box;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  if (o.stroke && o.stroke.length > 1) {
    ctx.beginPath();
    o.stroke.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.stroke();
  }
  ctx.setLineDash([]);
  // Points sampled from a scribble are implied by the stroke; only draw explicit ones.
  for (const p of o.points) if (!p.implicit) drawPoint(p, o.color);
}

function drawPoint(p, color) {
  const r = lineWidth() * 3;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
  ctx.fillStyle = p.label ? '#1db954' : '#e5383b';
  ctx.fill();
  ctx.lineWidth = lineWidth();
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.strokeStyle = '#fff';
  ctx.beginPath();
  ctx.moveTo(p.x - r * 0.5, p.y);
  ctx.lineTo(p.x + r * 0.5, p.y);
  if (p.label) {
    ctx.moveTo(p.x, p.y - r * 0.5);
    ctx.lineTo(p.x, p.y + r * 0.5);
  }
  ctx.stroke();
}

function drawDrag() {
  if (!drag) return;
  const color = COLORS[objectCounter % COLORS.length];
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth();
  if (drag.type === 'box') {
    const { start: a, end: b } = drag;
    ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
  } else {
    ctx.beginPath();
    drag.stroke.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.stroke();
  }
}

// ---------- UI ----------

function updateUI() {
  const pendingCount = objects.filter((o) => o.dirty).length;
  $('btn-capture').disabled = busy || cameraLoading;
  $('btn-flip').disabled = busy || cameraLoading || hasImage || !stream;
  $('btn-flip').hidden = !stream || hasImage;
  $('btn-upload').hidden = !!stream && !hasImage;
  $('btn-upload').disabled = busy || cameraLoading;
  $('file-input').disabled = busy || cameraLoading;
  $('btn-clear').disabled = busy || !hasImage;
  $('btn-segment').disabled = busy || !hasImage || !embeddingReady || pendingCount === 0;
  $('btn-segment').title = hasImage && !embeddingReady
    ? (embeddingFailed ? 'Image preparation failed' : 'Preparing image…') : '';
  const hasMasks = objects.some((o) => o.mask);
  $('btn-dl-labels').disabled = busy || !hasImage || !hasMasks;
}

function setBusy(text) {
  busy = !!text;
  $('busy').hidden = !text;
  $('busy-text').textContent = text ?? '';
  updateUI();
}

let toastTimer;
function toast(msg) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 4500);
}

function download(c, name) {
  c.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, 'image/png');
}

// Grayscale PNG where pixel value = object number (0 = background).
function downloadLabels() {
  const c = document.createElement('canvas');
  c.width = image.width;
  c.height = image.height;
  const cctx = c.getContext('2d');
  const out = cctx.createImageData(c.width, c.height);
  const px = out.data;
  for (let i = 0; i < px.length; i += 4) px[i + 3] = 255;
  objects.forEach((o, idx) => {
    if (!o.mask) return;
    const label = Math.min(255, idx + 1);
    for (let i = 0; i < o.mask.length; i++) {
      if (o.mask[i]) px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = label;
    }
  });
  cctx.putImageData(out, 0, 0);
  download(c, 'segmentation-labels.png');
}

// ---------- Wiring ----------

$('btn-flip').onclick = () => {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  startCamera();
};
$('btn-capture').onclick = () => stream ? capture() : startCamera();
$('btn-upload').onclick = () => $('file-input').click();
$('file-input').onchange = (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  upload(file);
};
$('btn-segment').onclick = segmentAll;
$('btn-clear').onclick = (e) => {
  const now = performance.now();
  if (e.detail === 2 || (lastClearTime && now - lastClearTime < 400)) {
    newPicture();
    return;
  }
  lastClearTime = now;
  objects = [];
  history = [];
  selectedId = null;
  drag = null;
  rebuildOverlay();
  afterEdit();
};
$('btn-dl-labels').onclick = downloadLabels;

document.addEventListener('keydown', (e) => {
  if (!busy && hasImage && (e.ctrlKey || e.metaKey) && e.key === 'z') {
    e.preventDefault();
    undo();
  }
});
