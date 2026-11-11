# Microplastic Bench

Detects microplastic particles in a live camera feed or an uploaded image, and classifies them as **fiber, film or fragment** with your trained YOLO11 model (`model/yolo11s.onnx`).

Everything runs in the browser. There is no backend and no build step.

## Project layout

```
microplastic-bench/
├── index.html                  Page markup (toolbar, viewer, count box)
├── css/style.css               Styling (dark theme, toolbar, viewer, phone layout)
├── js/app.js                   All logic: camera, image loading, rule-based detection,
│                               YOLO11 model loading and inference, drawing the overlay
├── model/yolo11s.onnx          Your YOLO11s model (fiber, film, fragment), exported to ONNX
├── vendor/onnxruntime-web/     ONNX Runtime Web 1.20.1 (runs the model offline)
├── server.js                   Tiny local web server (no dependencies)
├── package.json                `npm start` runs server.js
└── .vscode/                    Run/debug configs and recommended extensions
```

## Run it in VS Code

You need [Node.js](https://nodejs.org) (any recent version).

1. Open the `microplastic-bench` folder in VS Code (**File → Open Folder**).
2. Press **F5** and pick **Open in Chrome** (or **Open in Edge**). This starts the server and opens the app.

Or from the VS Code terminal:

```bash
npm start
```

then open http://localhost:5173.

The app must be opened through a server (http://localhost), not by double-clicking `index.html`. Browsers only allow the camera, the model file and the model runtime on `http://localhost` or `https://` addresses.

The **Live Server** extension also works: right-click `index.html` → **Open with Live Server**.

## Using it

- **Live**: detects particles continuously from the camera. Turns into **Stop**; **Freeze** keeps the current frame. A camera menu appears when more than one camera is connected (e.g. a USB microscope camera).
- **Upload**: analyse an image file. You can also drag and drop or paste an image.

The YOLO11 model loads automatically when the app opens. An image uploaded before it has finished loading is analysed as soon as it is ready.

## Updating the model

After retraining, export the new `best.pt` to ONNX and replace `model/yolo11s.onnx` with it (keep the same file name, or change `MODEL_URL` in `js/app.js`):

```bash
pip install ultralytics
yolo export model=best.pt format=onnx imgsz=640 opset=12
```

Class names and input size are read from the file.

## Settings to check (in `js/app.js`)

- `umPerPx: 3.5` in the `S` object: micrometres per pixel of your camera. Sizes are wrong until this matches your setup.
- `M.conf`: minimum confidence for a detection (0.25). Lower finds more particles but adds false detections; higher misses more.
- `M.iou`: overlap above which duplicate detections are merged (0.45).
- `NAMED`: colour for each class name.

## Limits

- Classification is by appearance only; it does not identify the polymer. Confirm with FTIR or Raman.
- Model validation scores (check results against hand counts):

  | Model | Classes | mAP50 | mAP50-95 | Precision | Recall |
  |---|---|---|---|---|---|
  | `yolo11s.onnx` | fiber, film, fragment | 0.52 | 0.34 | 0.53 | 0.58 |
