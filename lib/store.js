// History storage.
// - On Vercel (or anywhere DATABASE_URL / POSTGRES_URL is set): a Postgres table.
// - Locally without a database: a JSON file in data/history.json.
const fs = require("fs");
const path = require("path");

const DB_URL = process.env.DATABASE_URL || process.env.POSTGRES_URL || "";
const LIST_LIMIT = 200;

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
    ready = pool.query(`
      CREATE TABLE IF NOT EXISTS analyses (
        id          SERIAL PRIMARY KEY,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        image_name  TEXT,
        model       TEXT,
        um_per_px   REAL,
        total       INTEGER NOT NULL,
        counts      JSONB NOT NULL,
        particles   JSONB NOT NULL,
        thumb       TEXT NOT NULL,
        image       TEXT NOT NULL
      )`);
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
