# Microplastic Bench

Detects microplastic particles in a live camera feed or an uploaded image, and classifies them as **fiber, film or fragment** with your trained YOLO11 model (`model/yolo11s.onnx`).

Detection runs in the browser. A small backend (`api/history.js`) saves analyses to a **History**: on Vercel in a Postgres database, locally in a file. There is no build step.

## Project layout

```
microplastic-bench/
├── index.html                  Page markup (toolbar, viewer, count box)
├── css/style.css               Styling (dark theme, toolbar, viewer, phone layout)
├── js/app.js                   All logic: camera, image loading, rule-based detection,
│                               YOLO11 model loading and inference, drawing the overlay
├── model/yolo11s.onnx          Your YOLO11s model (fiber, film, fragment), exported to ONNX
├── vendor/onnxruntime-web/     ONNX Runtime Web 1.20.1 (runs the model offline; WebGPU + multi-core builds)
├── api/history.js              History API (save, list, open, delete), runs as a Vercel function
├── lib/store.js                History storage: Postgres when DATABASE_URL is set, else data/history.json
├── db/schema.sql               Database table for History (run once in Supabase's SQL Editor)
├── dev-server.js               Local server for development: serves the app and runs /api
├── package.json                `npm start` runs dev-server.js
├── vercel.json                 Vercel settings: static site + /api functions, no build step
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
- **Save**: stores the current result (annotated image, counts per class, and each particle's class, confidence, position and size) in the History. Shown when an image or frozen frame is analysed.
- **History**: lists saved analyses, newest first. Click one to view it; **Delete** (press twice) removes it.

The YOLO11 model loads automatically when the app opens. An image uploaded before it has finished loading is analysed as soon as it is ready.

## Live speed

- The camera picture is redrawn at the screen's frame rate; the outlines update each time the model finishes a frame, so they can trail fast movement slightly. The badge shows **detections/s**.
- The model runs on the **processor using several cores**. The browser console (F12) says which mode is used.
- Add `?gpu=1` to the address to try the **graphics card (WebGPU)**; on some laptops and phones it gives wrong or no results, so it is off by default.
- If detection fails or freezes, the app switches that browser to **compatibility mode** (one core). Force it with `?safe=1`; turn it off again with `?fast=1`.
- Multi-core needs the `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers; `vercel.json` and `dev-server.js` send them.
- For more speed, a YOLO11**n** model or exporting at a smaller size (`imgsz=480`) helps, at some cost in accuracy.

## History backend

| Where the app runs | Where history is stored |
|---|---|
| `npm start` on your computer | `data/history.json` in the project folder (not uploaded to GitHub) |
| Vercel | A **Supabase** Postgres database (any Postgres works, e.g. Neon) |

The API is `api/history.js`:

| Request | Does |
|---|---|
| `GET /api/history` | List saved analyses (newest 200, with thumbnails) |
| `GET /api/history?id=12` | One analysis, with the full annotated image and particle list |
| `POST /api/history` | Save an analysis |
| `DELETE /api/history?id=12` | Delete one |

### Access

History has **no passcode**: anyone who opens the app can save, view and delete analyses. Anyone with the link can do this, so share the link only with people who should use it.

Optional: to require a passcode later, add an `APP_PASSWORD` environment variable (in Vercel: **Settings → Environment Variables**). The app then asks for it once per browser.

### Set up the database: Supabase (one time)

1. Sign up at https://supabase.com and click **New project**. Pick a name, a **database password** (save it), and the region closest to you. Wait about a minute until the project is ready.
2. Click **Connect** (top of the project page) → **Connection String** → choose **Transaction pooler** (port 6543; best for Vercel). Copy the URI, which looks like:
   `postgresql://postgres.xxxxxxxx:[YOUR-PASSWORD]@aws-0-region.pooler.supabase.com:6543/postgres`
   Replace `[YOUR-PASSWORD]` with your database password. If the password contains `@ : / ? # %`, replace those characters with `%40 %3A %2F %3F %23 %25`.
3. Create the table: in Supabase open **SQL Editor** → **New query**, paste the contents of `db/schema.sql`, and click **Run**. (Optional: the app also creates it on first use.)
4. In Vercel: **Settings → Environment Variables** → add `DATABASE_URL` = the URI from step 2.
5. Redeploy (**Deployments** → **⋯** → **Redeploy**), or push a commit.

The `analyses` table has one row per saved analysis (see the comments in `db/schema.sql`). Row Level Security is switched on so Supabase's public REST API can't read it; only the app's `/api/history` can. To see your data, open the Supabase project → **Table Editor** → `analyses`.

Free plan notes: 500 MB of database storage, which holds roughly 2,000 saved analyses (each stores an annotated JPEG). Supabase pauses free projects after a week without use; open the Supabase dashboard and click **Restore** if History stops loading.

(Alternatively, Vercel's **Storage** tab can create a Supabase or Neon database and connect it for you; it sets `POSTGRES_URL`, which the app also reads.)

### Use a database locally (optional)

Run `npm install` once, then start with the database address, e.g. in PowerShell:

```powershell
$env:DATABASE_URL="postgres://user:password@host/dbname"; npm start
```

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
