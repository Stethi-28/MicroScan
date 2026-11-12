// History storage.
// - On Vercel (or anywhere DATABASE_URL / POSTGRES_URL is set): a Postgres table.
// - Locally without a database: a JSON file in data/history.json.
const fs = require("fs");
const path = require("path");

const DB_URL = process.env.DATABASE_URL || process.env.POSTGRES_URL || "";
const LIST_LIMIT = 200;

// Kept in sync with db/schema.sql (embedded so serverless deployments need no extra files).
const SCHEMA_SQL = `
-- MicroScan history database (PostgreSQL / Supabase)
-- Run once in Supabase: SQL Editor -> New query -> paste -> Run.
-- (The app also creates this table automatically on first use; running this script first is optional.)

CREATE TABLE IF NOT EXISTS analyses (
  id          SERIAL PRIMARY KEY,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  image_name  TEXT,                 -- uploaded file name, or "Live camera" / "Capture …"
  model       TEXT,                 -- model file used, e.g. yolo11s.onnx
  um_per_px   REAL,                 -- scale used for sizes (micrometres per pixel)
  total       INTEGER NOT NULL,     -- number of particles counted
  counts      JSONB   NOT NULL,     -- per class, e.g. {"Fiber": 3, "Film": 1, "Fragment": 5}
  particles   JSONB   NOT NULL,     -- list: id, class, confidence, x_px, y_px, length_um, width_um, area_um2
  thumb       TEXT    NOT NULL,     -- small annotated JPEG (data URL) for the History list
  image       TEXT    NOT NULL      -- annotated JPEG (data URL), up to 1280 px
);

-- Newest-first listing used by the History panel.
CREATE INDEX IF NOT EXISTS analyses_created_at_idx ON analyses (created_at DESC);

-- Supabase also exposes tables through its public REST API. Row Level Security with no policies
-- blocks that route, so the data is only reachable through the app's own /api/history.
-- (The app connects as the table owner, which RLS does not restrict.)
ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;
`;

/* ---------------- Postgres ---------------- */
function withoutSslParams(url) {
  try {
    const u = new URL(url);
    for (const k of ["sslmode", "sslcert", "sslkey", "sslrootcert", "uselibpqcompat", "supa"]) u.searchParams.delete(k);
    return u.toString();
  } catch (e) {
    return url;
  }
}
let pool = null;
let ready = null;
function pg() {
  if (!pool) {
    const { Pool } = require("pg");
    const local = /@(localhost|127\.0\.0\.1)[:/]/.test(DB_URL);
    // Hosted databases (Supabase, Neon) add "?sslmode=require" to their URLs. The pg driver would then
    // insist on verifying the provider's own certificate authority and fail, so the SSL options are set here instead.
    pool = new Pool({ connectionString: withoutSslParams(DB_URL), ssl: local ? false : { rejectUnauthorized: false }, max: 3 });
    // Create the table, index and Row Level Security if they don't exist yet (same as db/schema.sql).
    ready = pool.query(SCHEMA_SQL);
  }
  return ready.then(() => pool);
}

const pgStore = {
  async list() {
    const db = await pg();
    const r = await db.query(
      `SELECT id, created_at, image_name, model, total, counts, thumb
         FROM analyses ORDER BY created_at DESC, id DESC LIMIT $1`, [LIST_LIMIT]);
    return r.rows;
  },
  async get(id) {
    const db = await pg();
    const r = await db.query(`SELECT * FROM analyses WHERE id = $1`, [id]);
    return r.rows[0] || null;
  },
  async add(a) {
    const db = await pg();
    const r = await db.query(
      `INSERT INTO analyses (image_name, model, um_per_px, total, counts, particles, thumb, image)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id, created_at`,
      [a.image_name, a.model, a.um_per_px, a.total, JSON.stringify(a.counts), JSON.stringify(a.particles), a.thumb, a.image]);
    return r.rows[0];
  },
  async remove(id) {
    const db = await pg();
    const r = await db.query(`DELETE FROM analyses WHERE id = $1`, [id]);
    return r.rowCount > 0;
  },
};

/* ---------------- JSON file (local development) ---------------- */
const FILE = path.join(__dirname, "..", "data", "history.json");
function readAll() {
  try { return JSON.parse(fs.readFileSync(FILE, "utf8")); } catch (e) { return []; }
}
function writeAll(rows) {
  fs.mkdirSync(path.dirname(FILE), { recursive: true });
  const tmp = FILE + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(rows));
  fs.renameSync(tmp, FILE);
}
const fileStore = {
  async list() {
    return readAll()
      .sort((a, b) => b.id - a.id)
      .slice(0, LIST_LIMIT)
      .map(({ id, created_at, image_name, model, total, counts, thumb }) => ({ id, created_at, image_name, model, total, counts, thumb }));
  },
  async get(id) { return readAll().find((r) => r.id === id) || null; },
  async add(a) {
    const rows = readAll();
    const id = rows.reduce((m, r) => Math.max(m, r.id), 0) + 1;
    const row = { id, created_at: new Date().toISOString(), ...a };
    rows.push(row);
    writeAll(rows);
    return { id, created_at: row.created_at };
  },
  async remove(id) {
    const rows = readAll();
    const left = rows.filter((r) => r.id !== id);
    writeAll(left);
    return left.length < rows.length;
  },
};

function store() {
  if (DB_URL) return pgStore;
  if (process.env.VERCEL) {
    throw new Error("No database is connected. In Vercel, add the DATABASE_URL environment variable with your Supabase connection string (Settings → Environment Variables), then redeploy.");
  }
  return fileStore;
}

module.exports = { store, usingDatabase: !!DB_URL };
