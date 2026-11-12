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
