// History API
//   GET    /api/history          -> { items: [ {id, created_at, image_name, model, total, counts, thumb} ] }
//   GET    /api/history?id=12    -> full record (includes annotated image and particle list)
//   POST   /api/history          -> save an analysis, returns { id, created_at }
//   DELETE /api/history?id=12    -> delete one
// Open to anyone using the app. Optionally protected by the APP_PASSWORD environment variable (x-app-key header).
const crypto = require("crypto");
const { store } = require("../lib/store");

const MAX_IMAGE = 3 * 1024 * 1024; // characters of base64 data URL
const MAX_THUMB = 300 * 1024;
const DATA_URL = /^data:image\/(jpeg|png|webp);base64,[A-Za-z0-9+/=]+$/;

function sameKey(given, expected) {
  const a = Buffer.from(String(given || ""));
  const b = Buffer.from(String(expected));
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

function validate(b) {
  if (!b || typeof b !== "object") return "Request body must be JSON.";
  if (typeof b.image !== "string" || b.image.length > MAX_IMAGE || !DATA_URL.test(b.image)) return "image must be a JPEG/PNG data URL under 3 MB.";
  if (typeof b.thumb !== "string" || b.thumb.length > MAX_THUMB || !DATA_URL.test(b.thumb)) return "thumb must be a small JPEG/PNG data URL.";
  if (!Number.isInteger(b.total) || b.total < 0) return "total must be a whole number.";
  if (!b.counts || typeof b.counts !== "object" || Array.isArray(b.counts) ||
      !Object.values(b.counts).every((v) => Number.isInteger(v) && v >= 0)) return "counts must map class names to whole numbers.";
  if (!Array.isArray(b.particles) || b.particles.length > 5000) return "particles must be a list (at most 5000).";
  return null;
}

function clean(b) {
  return {
    image_name: String(b.image_name || "Image").slice(0, 200),
    model: String(b.model || "").slice(0, 100),
    um_per_px: Number.isFinite(+b.um_per_px) ? +b.um_per_px : null,
    total: b.total,
    counts: b.counts,
    particles: b.particles,
    thumb: b.thumb,
    image: b.image,
  };
}

module.exports = async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  try {
    // No passcode by default: anyone who opens the app can use History.
    // Optional: set APP_PASSWORD to require a passcode.
    const key = process.env.APP_PASSWORD;
    if (key && !sameKey(req.headers["x-app-key"], key)) {
      return res.status(401).json({ error: "Passcode required." });
    }

    const url = new URL(req.url, "http://localhost");
    const idParam = (req.query && req.query.id) || url.searchParams.get("id");
    const id = idParam != null ? Number.parseInt(idParam, 10) : null;
    if (idParam != null && !(id > 0)) return res.status(400).json({ error: "Invalid id." });

    const db = store();

    if (req.method === "GET") {
      if (id) {
        const item = await db.get(id);
        return item ? res.status(200).json(item) : res.status(404).json({ error: "Not found." });
      }
      return res.status(200).json({ items: await db.list() });
    }

    if (req.method === "POST") {
      let body = req.body;
      if (typeof body === "string") { try { body = JSON.parse(body); } catch (e) { body = null; } }
      const err = validate(body);
      if (err) return res.status(400).json({ error: err });
      const saved = await db.add(clean(body));
      return res.status(201).json(saved);
    }

    if (req.method === "DELETE") {
      if (!id) return res.status(400).json({ error: "Missing id." });
      const ok = await db.remove(id);
      return ok ? res.status(200).json({ ok: true }) : res.status(404).json({ error: "Not found." });
    }

    res.setHeader("Allow", "GET, POST, DELETE");
    return res.status(405).json({ error: "Method not allowed." });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: e.message || "Server error." });
  }
};
