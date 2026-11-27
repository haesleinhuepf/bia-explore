# Snap & Segment

Take a photo with your laptop or phone camera, annotate the objects you care about, and segment them with
[SAM 2.1](https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX). Everything runs in the browser
using [transformers.js](https://github.com/huggingface/transformers.js). There is no backend.

![Mobile view of Snap & Segment with an example image and segmentation](screenshot.png)

## Run

The files are static, so any static web server works:

```
npx serve .
```

Then open `http://localhost:3000`.

Browsers only allow camera access in a secure context: `localhost` or HTTPS. To use the camera on a phone,
host the folder somewhere with HTTPS, for example GitHub Pages. Without HTTPS, **Upload image** still works
on phones and also offers the camera app.

The first start downloads the model, about 90 MB with WebGPU or 155 MB without it. The browser caches the
model, so later starts skip the download.

## Use

1. **Start camera** → **Take picture** (or **Upload image**).
2. Mark each object:
   - **Box**: drag a box around an object. This creates a new object.
   - **Scribble**: draw on an object. This creates a new object, and points sampled along the stroke become
     prompts.
   - **＋ / − Point**: refine the selected object with include and exclude clicks.
3. **Segment objects**. Only new or changed objects are recomputed.
4. Download the colored overlay, or a label image where each pixel value is the object number.

The side panel scrolls independently of the image. Use the arrow at the top to move it to the other side.
Segmentation becomes available after the image is prepared and at least one object has been marked.

## How it works

- `worker.js` loads `Sam2Model` in a Web Worker. It uses WebGPU (fp16 encoder) when available and falls back
  to WASM (fp32).
- The image encoder runs once, right after capture. Each object then needs only a fast prompt
  encoder/mask decoder pass, using box and/or point prompts.
- SAM returns three candidate masks per object. The app keeps the one with the highest predicted IoU.
