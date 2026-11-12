// Local development server (no dependencies).
// Usage: npm start  ->  http://localhost:5173
// Serves the app's files and runs the /api functions the same way Vercel does.
// Named dev-server.js (not server.js) so Vercel deploys the app as a static site with /api functions.
const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = Number(process.env.PORT) || 5173;
const ROOT = __dirname;
const MAX_BODY = 10 * 1024 * 1024;
const PRIVATE = /^\/(api|lib|data|node_modules|\.git|\.vscode)(\/|$)|^\/(dev-server\.js|package(-lock)?\.json|vercel\.json|\.env.*)$/;
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

// Minimal Vercel-style req/res helpers for the API handlers.
function runApi(handlerFile, req, res, urlPath) {
  const chunks = [];
  let size = 0;
  req.on("data", (c) => {
    size += c.length;
    if (size > MAX_BODY) { res.writeHead(413); res.end("Too large"); req.destroy(); }
    else chunks.push(c);
  });
  req.on("end", async () => {
    if (res.writableEnded) return;
    const raw = Buffer.concat(chunks).toString("utf8");
    const url = new URL(req.url, "http://localhost");
    req.query = Object.fromEntries(url.searchParams);
    try { req.body = raw && /json/.test(req.headers["content-type"] || "") ? JSON.parse(raw) : raw || undefined; }
    catch (e) { req.body = raw; }
    res.status = (code) => { res.statusCode = code; return res; };
    res.json = (obj) => {
      if (!res.getHeader("Content-Type")) res.setHeader("Content-Type", "application/json; charset=utf-8");
      res.end(JSON.stringify(obj));
      return res;
    };
    try {
      await require(handlerFile)(req, res);
    } catch (e) {
      console.error(e);
      if (!res.writableEnded) res.status(500).json({ error: "Server error." });
    }
  });
}

http
  .createServer((req, res) => {
    let urlPath = decodeURIComponent(req.url.split("?")[0]);

    // API functions: /api/<name> -> api/<name>.js
    const api = urlPath.match(/^\/api\/([\w-]+)\/?$/);
    if (api) {
      const file = path.join(ROOT, "api", api[1] + ".js");
      if (fs.existsSync(file)) return runApi(file, req, res, urlPath);
      res.writeHead(404);
      return res.end("Not found");
    }

    if (urlPath.endsWith("/")) urlPath += "index.html";
    if (PRIVATE.test(urlPath)) {
      res.writeHead(404);
      return res.end("Not found");
    }
    const file = path.normalize(path.join(ROOT, urlPath));
    if (!file.startsWith(ROOT + path.sep)) {
      res.writeHead(403);
      return res.end("Forbidden");
    }
    fs.readFile(file, (err, data) => {
      if (err) {
        res.writeHead(404);
        return res.end("Not found");
      }
      res.writeHead(200, {
        "Content-Type": TYPES[path.extname(file).toLowerCase()] || "application/octet-stream",
        "Cache-Control": "no-cache",
        "Content-Length": data.length,
        // Cross-origin isolation lets the model use several processor cores (same headers as vercel.json).
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "credentialless",
      });
      res.end(data);
    });
  })
  .listen(PORT, () => {
    const { usingDatabase } = require("./lib/store");
    console.log(`Microplastic Bench running at http://localhost:${PORT}`);
    console.log(usingDatabase ? "History: saved in the Postgres database (DATABASE_URL)." : "History: saved locally in data/history.json.");
    if (process.env.APP_PASSWORD) console.log("History is protected by APP_PASSWORD.");
  });
