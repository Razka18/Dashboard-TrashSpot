"""Penyimpanan riwayat pemeriksaan TrashSpot (SQLite).

Satu baris `analyses` = satu gambar yang diambil dari sebuah CCTV lalu
dianalisis YOLO (otomatis berkala atau manual lewat tombol).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "data" / "trashspot.db"

CATEGORIES = ["Bersih", "Cukup Bersih", "Perlu Dibersihkan", "Sangat Perlu Dibersihkan"]
PRIORITY = {"Bersih": "Rendah", "Cukup Bersih": "Rendah",
            "Perlu Dibersihkan": "Sedang", "Sangat Perlu Dibersihkan": "Tinggi"}
# Ambang awal (jumlah objek per gambar) - kalibrasi bersama petugas desa.
DEFAULT_THRESHOLDS = {"bersih": 4, "cukup": 12, "perlu": 24}

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_no INTEGER NOT NULL,
    camera_label TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    source_video TEXT,
    frame_idx INTEGER,
    photo_path TEXT,
    thumb_path TEXT,
    total INTEGER NOT NULL,
    category TEXT NOT NULL,
    priority TEXT NOT NULL,
    counts_json TEXT NOT NULL,
    after_cleaning INTEGER NOT NULL DEFAULT 0,
    trigger TEXT NOT NULL DEFAULT 'auto',
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_an_cam ON analyses(camera_no, captured_at);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as c:
        c.executescript(SCHEMA)


# ------------------------------------------------------------------ ambang
def get_thresholds() -> dict[str, int]:
    with _conn() as c:
        rows = dict(c.execute("SELECT key, value FROM settings").fetchall())
    return {k: int(rows.get(f"t_{k}", v)) for k, v in DEFAULT_THRESHOLDS.items()}


def set_thresholds(bersih: int, cukup: int, perlu: int) -> None:
    with _conn() as c:
        c.executemany("INSERT OR REPLACE INTO settings VALUES (?,?)",
                      [("t_bersih", str(bersih)), ("t_cukup", str(cukup)),
                       ("t_perlu", str(perlu))])


def categorize(total: int, th: dict[str, int]) -> str:
    if total <= th["bersih"]:
        return CATEGORIES[0]
    if total <= th["cukup"]:
        return CATEGORIES[1]
    if total <= th["perlu"]:
        return CATEGORIES[2]
    return CATEGORIES[3]


# ------------------------------------------------------------------ analisis
def add_analysis(*, camera_no: int, camera_label: str, captured_at: datetime,
                 source_video: str, frame_idx: int, photo_path: str, thumb_path: str,
                 total: int, category: str, counts: dict[str, int],
                 after_cleaning: bool, trigger: str, model_version: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO analyses (camera_no, camera_label, captured_at, source_video,"
            " frame_idx, photo_path, thumb_path, total, category, priority, counts_json,"
            " after_cleaning, trigger, model_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (camera_no, camera_label, captured_at.strftime("%Y-%m-%d %H:%M:%S"),
             source_video, frame_idx, photo_path, thumb_path, total, category,
             PRIORITY[category], json.dumps(counts, ensure_ascii=False),
             int(after_cleaning), trigger, model_version))
        return int(cur.lastrowid)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df["captured_at"] = pd.to_datetime(df["captured_at"])
    df["counts"] = df["counts_json"].apply(json.loads)
    return df


def get_analyses(camera_no: int | None = None, limit: int | None = None) -> pd.DataFrame:
    """Terbaru lebih dulu."""
    sql, params = "SELECT * FROM analyses", []
    if camera_no is not None:
        sql += " WHERE camera_no=?"
        params.append(camera_no)
    sql += " ORDER BY captured_at DESC, id DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with _conn() as c:
        return _prep(pd.read_sql_query(sql, c, params=params))


def latest_per_camera() -> pd.DataFrame:
    """Pemeriksaan terbaru tiap kamera, dengan urutan yang sama seperti halaman Riwayat
    (captured_at lalu id), supaya Dashboard dan Riwayat selalu menunjuk baris yang sama."""
    sql = ("SELECT a.* FROM analyses a WHERE a.id = (SELECT b.id FROM analyses b "
           "WHERE b.camera_no = a.camera_no ORDER BY b.captured_at DESC, b.id DESC LIMIT 1) "
           "ORDER BY a.camera_no")
    with _conn() as c:
        return _prep(pd.read_sql_query(sql, c))


def get_analysis(analysis_id: int) -> dict | None:
    with _conn() as c:
        df = pd.read_sql_query("SELECT * FROM analyses WHERE id=?", c, params=[analysis_id])
    return None if df.empty else _prep(df).iloc[0].to_dict()