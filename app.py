"""TrashSpot — pemantauan kebersihan desa dari CCTV dengan YOLO26s.

Jalankan dari folder proyek:  streamlit run app.py

Path video diisi manual di CAMERA_CONFIG (lihat bagian bawah import).
Model YOLO dibaca otomatis dari MODEL/best (2).pt
"""
from __future__ import annotations

import base64
import html
import logging
import os
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# Cegah crash FFmpeg "async_lock" (harus diset sebelum cv2 diimpor)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")

import cv2
import numpy as np
import streamlit as st

# Windows: tab browser ditutup/refresh memicu "ConnectionResetError WinError 10054" dari asyncio.
# Itu hanya log berisik (bukan error aplikasi), jadi disembunyikan.
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

import database as db
from cctv import CameraPlayer, build_cameras
from detector import WasteDetector, draw_detections, find_model, pretty

BASE = Path(__file__).resolve().parent
_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")  # deteksi di thread terpisah
THUMB_EVERY = 0.6    # detik: kecepatan segar thumbnail kamera non-aktif
INFER_EVERY = 0.35   # detik: jeda minimal antar deteksi YOLO pada kamera terpilih

# =====================================================================
#  ISI PATH VIDEO CCTV DI SINI
# ---------------------------------------------------------------------
#  - "videos" boleh berisi file video, folder (semua video di dalamnya
#    diputar berurutan), atau campuran keduanya.
#  - Windows: awali string dengan r  ->  r"D:\Data\CCTV\jalan.mp4"
#    (atau pakai garis miring biasa  ->  "D:/Data/CCTV/jalan.mp4")
#  - Path relatif dihitung dari folder app.py.
#  - Kamera yang tidak diisi / path-nya salah akan tampil "offline".
# =====================================================================
CAMERA_CONFIG = {
    1: {"label": "Jalan Utama", "banjar": "Banjar Dinas Delod Rurung",
        "videos": ["CCTV_1_Jalan_Utama"]},
    2: {"label": "Taman dan Ruang Terbuka", "banjar": "Banjar Dinas Gerokgak Gede",
        "videos": ["CCTV_2_Taman_dan_Ruang_Hijau"]},   # <- lengkapi nama foldernya
    3: {"label": "Kantor Desa", "banjar": "Banjar Dinas Gerokgak Tengah",
        "videos": ["PERLU_VERIFIKASI_Tidak_Sesuai_Kategori"]},                              # tidak ada folder = offline
    4: {"label": "Tempat Sampah", "banjar": "Banjar Dinas Sakenan Blodan",
        "videos": ["CCTV_4_Tempat_Sampah"]},
}
CAPTURE_DIR = BASE / "data" / "captures"
MAX_TICKS = int(os.environ.get("TRASHSPOT_MAX_TICKS", "0"))  # 0 = tanpa batas (untuk uji)

st.set_page_config(page_title="TrashSpot", page_icon="🗑️", layout="wide",
                   initial_sidebar_state="collapsed")
db.init_db()
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

PAGES = ["Dashboard", "Pantau CCTV", "Riwayat"]
CATS = {  # kategori: (latar, teks, titik/bar)
    "Bersih": ("#dcebe3", "#1f5a41", "#2d6a4f"),
    "Cukup Bersih": ("#eaf1d3", "#4a5e12", "#8fa83a"),
    "Perlu Dibersihkan": ("#fde7c2", "#7a4a00", "#e0a21c"),
    "Sangat Perlu Dibersihkan": ("#f8d3c5", "#8a2a0a", "#c4410c"),
}
# Ambang kategori: jumlah objek sampah terlihat per foto (sesuai tabel laporan).
#   0 -> Bersih | 1-3 -> Cukup Bersih | 4-9 -> Perlu Dibersihkan | >=10 -> Sangat Perlu Dibersihkan
CATEGORY_RULES = ((0, "Bersih"), (3, "Cukup Bersih"), (9, "Perlu Dibersihkan"))
CATEGORY_TOP = "Sangat Perlu Dibersihkan"


def categorize(n: int, th=None) -> str:
    """Kategori kebersihan dari jumlah objek per foto. Parameter `th` diabaikan
    (dipertahankan agar pemanggilan lama tetap jalan)."""
    for upper, name in CATEGORY_RULES:
        if n <= upper:
            return name
    return CATEGORY_TOP


# Data lama di database menyimpan kategori dari ambang lama. Supaya semua tampilan selalu
# konsisten dengan tabel ambang di atas, kategori dihitung ulang dari jumlah objek (`total`).
def _recat(df):
    if df is not None and len(df) and "total" in df.columns:
        df = df.copy()
        df["category"] = df["total"].astype(int).map(categorize)
    return df


def _latest_per_camera():
    return _recat(db.latest_per_camera())


def _get_analyses(*args, **kwargs):
    return _recat(db.get_analyses(*args, **kwargs))


def _get_analysis(analysis_id):
    r = db.get_analysis(analysis_id)
    if r is not None and r.get("total") is not None:
        r = dict(r)
        r["category"] = categorize(int(r["total"]))
    return r


INTERVALS = {"30 detik": 30, "1 menit": 60, "5 menit": 300, "10 menit": 600,
             "30 menit": 1800, "Manual saja": None}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {color-scheme:light; --bg:#f1f4ec; --ink:#1c2b21; --muted:#5d6b61; --line:#e0e6d6; --green:#2d6a4f; --gap:24px;}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background: var(--bg); color: var(--ink);}
html, body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"],
.stMain, section.main {overflow-x:hidden !important; max-width:100vw;}
.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp li {font-family: 'Inter', sans-serif;}
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
#MainMenu, footer {display: none !important;}
.block-container {max-width: 1320px; padding: .6rem 2.5rem 3rem !important;}
h1, h2, h3 {font-family: 'Bricolage Grotesque', sans-serif !important; color: var(--ink);}
[data-testid="stVerticalBlock"] {gap: 1rem;}
.st-key-page {gap: var(--gap) !important;}          /* jarak antar bagian halaman */
/* samakan margin bawaan Streamlit agar semua jarak murni berasal dari `gap` */
.st-key-page [data-testid="stElementContainer"], .st-key-page [data-testid="stMarkdownContainer"] {
    margin-top:0 !important; margin-bottom:0 !important;}

/* ---- komponen bawaan Streamlit diselaraskan dengan desain ---- */
[data-baseweb="select"] > div {min-height:44px;}
.pillrow {height:44px; display:flex; align-items:center;}
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input {background:#fff !important;
    border-color:#cdd5c0 !important; border-radius:10px !important;}
[data-baseweb="select"] *, [data-baseweb="select"] input {color:#1c2b21 !important; -webkit-text-fill-color:#1c2b21 !important;}
[data-testid="stCheckbox"] p, [data-testid="stExpander"] summary p {color:#1c2b21 !important;}
[data-testid="stWidgetLabel"] p {font-size:13px; font-weight:600; color:var(--ink);}
[data-testid="stExpander"] {border:1px solid var(--line); border-radius:12px; background:#fafbf7;}
[data-testid="stCaptionContainer"] p {color:var(--muted); font-size:12.5px; line-height:1.5;}
[data-testid="stElementToolbar"] {display:none !important;}
.st-key-js_host {position:absolute; width:0; height:0; overflow:hidden; margin:0 !important;}

/* ---- top bar ---- */
.st-key-topbar {position:relative; padding:.9rem 0 1rem; margin-bottom:1.2rem;}
.st-key-topbar::after {content:""; position:absolute; bottom:0; left:50%; width:100vw; margin-left:-50vw;
    border-bottom:1px solid #d8dfcc; pointer-events:none;}
.logo {display:flex; align-items:center; gap:12px; font-family:'Bricolage Grotesque',sans-serif;
    font-weight:700; font-size:26px; color:var(--ink);}
.loc-pill {text-align:right;}
.loc-pill span {display:inline-block; border:1.5px solid var(--ink); border-radius:999px;
    padding:.5rem 1.1rem; font-family:'JetBrains Mono',monospace; font-size:13px; white-space:nowrap;}
.st-key-topbar [data-testid="stButton"] {display:flex; justify-content:center;}
.st-key-topbar button {width:auto !important; background:transparent !important; border:none !important; box-shadow:none !important;
    border-bottom:2px solid transparent !important; border-radius:0 !important; color:var(--ink) !important;
    padding:.35rem .1rem; min-height:0; transition: color .18s, transform .18s, border-color .18s;}
.st-key-topbar button p {font-size:15px; font-weight:500;}
.st-key-topbar button:hover {color:var(--green) !important; transform:translateY(-1px);
    border-bottom-color:rgba(45,106,79,.45) !important;}
.st-key-topbar [data-testid="stBaseButton-primary"] {border-bottom:2px solid var(--green) !important;}

/* ---- judul halaman ---- */
.page-head {display:flex; justify-content:space-between; align-items:flex-end; margin:0; gap:1rem;}
.page-head h1 {font-weight:800; font-size:44px; line-height:1.1; margin:0 0 .35rem; padding:0;}
.page-head p {color:var(--muted); font-size:16px; margin:0;}
.pill-online {display:inline-flex; align-items:center; gap:8px; background:#dcebe3; color:#1f5a41;
    border-radius:999px; padding:.4rem 1rem; font-size:13.5px; white-space:nowrap;}
.pill-online i {width:9px; height:9px; border-radius:50%; background:#2d6a4f; display:inline-block;}
.mono-pill {background:#e2e8d8; border-radius:8px; padding:.45rem .9rem; color:#4a5a4f;
    font-family:'JetBrains Mono',monospace; font-size:13px; white-space:nowrap;}

/* ---- kartu & pil ---- */
.card, .st-key-auto_card, .st-key-prio_card, .st-key-hist_list {background:#fff; border-radius:18px;
    padding:22px 24px; box-shadow:0 1px 2px rgba(30,60,40,.05); box-sizing:border-box;}
.st-key-auto_card, .st-key-prio_card, .st-key-hist_list {gap:.7rem;}
.card h3, .st-key-auto_card h3, .st-key-prio_card h3, .st-key-hist_list h3 {font-size:21px; margin:0; padding:0;}
.card h3 {margin-bottom:.8rem;}
/* kartu = container ber-key "card_*"; antar kartu selalu berjarak --gap dari container induknya */
[class*="st-key-card_"] {background:#fff; border-radius:18px; padding:22px 24px; box-sizing:border-box;
    box-shadow:0 1px 2px rgba(30,60,40,.05); gap:0 !important;}
[class*="st-key-card_"] h3 {font-size:21px; margin:0 0 .8rem; padding:0;}
.st-key-dash_left {gap:var(--gap) !important;}
.muted-sm {color:var(--muted); font-size:13px; margin-bottom:.5rem;}
.pill {display:inline-flex; align-items:center; gap:8px; border-radius:999px; padding:.3rem .9rem;
    font-weight:500; font-size:13.5px; white-space:nowrap;}
.pill i {width:9px; height:9px; border-radius:50%; display:inline-block; flex:none;}
.pill.lg {font-size:19px; padding:.55rem 1.2rem; margin-bottom:.9rem;}
.pill.wide {display:flex; width:100%; border-radius:10px; padding:.42rem .8rem; box-sizing:border-box;}
.pill.muted {background:#eceee6; color:#6b786f;}
.kv {display:flex; justify-content:space-between; font-size:14px; padding:.25rem 0;}
.kv b {font-weight:600;}
.trow-c {display:flex; justify-content:space-between; padding:.8rem 0; border-bottom:1px solid var(--line); font-size:15px;}
.trow-c:last-child {border-bottom:none; padding-bottom:0;}
.trow-c b {font-weight:600;}

/* ---- kartu kamera: seluruh kartu bisa diklik ---- */
[class*="st-key-cam_"] {position:relative; background:#fff; border:2px solid var(--line); border-radius:18px;
    padding:12px 12px 14px; gap:.3rem; min-height:190px; box-sizing:border-box;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;}
[class*="st-key-cam_"]:hover {transform:translateY(-3px); box-shadow:0 10px 24px rgba(30,60,40,.13); border-color:#9bb8a6;}
[class*="st-key-cam_"] [data-testid="stImage"] img {width:100%; height:86px; object-fit:cover; border-radius:10px;}
[class*="st-key-cam_"] [class*="st-key-pick_"] {position:absolute !important; inset:0; z-index:5;
    width:100% !important; height:100% !important; margin:0 !important;}
[class*="st-key-cam_"] [class*="st-key-pick_"] [data-testid="stButton"] {width:100%; height:100%;}
[class*="st-key-cam_"] [class*="st-key-pick_"] button {width:100%; height:100%; min-height:0; opacity:0;
    cursor:pointer; border-radius:18px;}
.thumb-ph {height:86px; border-radius:10px; background:#3b463f;}
.thumb-ph.hatch {background:repeating-linear-gradient(135deg,#d3d7c9 0 8px,#e2e5d9 8px 16px);}
.cam-title {font-weight:600; font-size:15px; margin-top:8px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cam-sub {font-size:12.5px; color:var(--muted); margin:2px 0 10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cam-off {font-size:13px; padding:.4rem 0; color:var(--muted);}

/* ---- pantau cctv: video kiri, panel kanan ---- */
.st-key-pantau_row {display:grid !important; grid-template-columns:minmax(0,1.42fr) minmax(0,1fr);
    gap:var(--gap) !important; align-items:start;}
.st-key-pv_left {gap:16px !important;}
.st-key-pad_a, .st-key-pad_b, .st-key-pad_c {display:none !important;}
.st-key-pv_right {gap:var(--gap) !important;}
.types-scroll {min-height:96px; max-height:240px; overflow-y:auto; padding-right:4px;}
.st-key-main_card [data-testid="stImage"] img {width:100%; border-radius:18px;}
.main-skel {height:420px; border-radius:18px; background:#3b463f; color:#cfd8cf; display:flex;
    align-items:center; justify-content:center; font-family:'JetBrains Mono',monospace;}
.st-key-btn_capture button, .st-key-btn_check button {background:var(--green); color:#fff; border:none;
    border-radius:10px; padding:.65rem 1.3rem; font-weight:600; transition: transform .18s, box-shadow .18s, background .18s;}
.st-key-btn_capture button:hover, .st-key-btn_check button:hover {background:#357f5e; color:#fff;
    transform:translateY(-2px); box-shadow:0 8px 18px rgba(45,106,79,.3);}
.st-key-btn_capture button p, .st-key-btn_check button p {color:#fff;}
.st-key-btn_hist button, .st-key-btn_more button {background:#fff; border:1.5px solid var(--ink); color:var(--ink);
    border-radius:10px; padding:.65rem 1.3rem; font-weight:600; transition: transform .18s, background .18s;}
.st-key-btn_hist button:hover, .st-key-btn_more button:hover {background:#e9eee0; border-color:var(--ink);
    color:var(--ink); transform:translateY(-2px);}

/* ---- dashboard ---- */
.kpi-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:var(--gap);}
.kpi {background:#fff; border-radius:16px; padding:20px 22px; min-height:100px; box-sizing:border-box;
    display:flex; flex-direction:column; justify-content:center; box-shadow:0 1px 2px rgba(30,60,40,.05);}
.kpi b {font-family:'Bricolage Grotesque',sans-serif; font-size:34px; font-weight:700; display:block; line-height:1.2;}
.kpi b.sm {font-size:30px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.kpi span {font-size:13px; color:var(--muted);}
.kpi.alert {background:#f8d3c5;} .kpi.alert b {color:#8a2a0a;} .kpi.alert span {color:#8a2a0a;}
.st-key-dash_row {display:grid !important; grid-template-columns:minmax(0,1fr) minmax(0,1.25fr);
    gap:var(--gap) !important; align-items:start;}
.bar {height:10px; border-radius:99px; background:#e0e6d6; overflow:hidden;}
.bar > span {display:block; height:100%; border-radius:99px;}
.lvl {margin-bottom:1.15rem;} .lvl:last-child {margin-bottom:0;}
.lvl-top {display:flex; justify-content:space-between; align-items:center; margin-bottom:.45rem;}
.trow {display:grid; grid-template-columns:120px minmax(0,1fr) 36px; align-items:center; gap:14px; padding:.45rem 0; font-size:14.5px;}
.trow span:first-child {overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
.trow b {text-align:right;}
.prio-row {display:grid; grid-template-columns:28px minmax(0,1fr) auto 84px; align-items:center; gap:12px;
    padding:14px 0; border-top:1px solid var(--line);}
.prio-row .n {font-weight:700;} .prio-row .nm {font-weight:600;}
.prio-row .nm, .prio-row .sb {overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
.prio-row .sb {font-size:12.5px; color:var(--muted);}
.prio-row .it {font-family:'JetBrains Mono',monospace; font-size:14px; text-align:right; white-space:nowrap;}

/* ---- riwayat ---- */
.chart {display:flex; align-items:flex-end; gap:26px; overflow-x:auto; padding:8px 4px 4px;}
.chart .col {flex:1; min-width:70px; display:flex; flex-direction:column; align-items:center;}
.chart .plot {height:240px; display:flex; flex-direction:column; justify-content:flex-end; align-items:center;}
.chart .plot b {margin-bottom:6px; font-size:14px;}
.chart .b {width:64px; border-radius:6px 6px 0 0; transition: filter .18s;} .chart .b:hover {filter:brightness(1.08);}
.chart .xl {font-size:12.5px; color:var(--muted); text-align:center; margin-top:8px; line-height:1.35;}
[class*="st-key-hrow_"] {border-top:1px solid var(--line); padding:12px 0 0; gap:0;}
[class*="st-key-hrow_"] p {margin:0; font-size:14.5px;}
[class*="st-key-hrow_"] [data-testid="stImage"] img {width:56px; height:40px; object-fit:cover; border-radius:8px;
    transition: transform .18s, box-shadow .18s;}
/* thumbnail = tombol "perbesar" transparan (klik gambar untuk memperbesar) */
[class*="st-key-zc_"] {position:relative; width:56px; height:40px; gap:0;}
[class*="st-key-zc_"]:hover img {transform:scale(1.08); box-shadow:0 6px 14px rgba(30,60,40,.25);}
[class*="st-key-zc_"] [class*="st-key-zb_"] {position:absolute !important; inset:0; z-index:5;
    width:100% !important; height:100% !important; margin:0 !important;}
[class*="st-key-zc_"] [class*="st-key-zb_"] [data-testid="stButton"] {width:100%; height:100%;}
[class*="st-key-zc_"] [class*="st-key-zb_"] button {width:100%; height:100%; min-height:0; padding:0; opacity:0; cursor:zoom-in;}
.r {text-align:right;}
.dt {font-family:'JetBrains Mono',monospace; font-size:13px;}
.ph {width:56px; height:40px; border-radius:8px; background:#4b5148;}

@media (max-width: 900px) {
    .block-container {padding: .5rem 1rem 2rem !important;}
    .page-head {flex-direction:column; align-items:flex-start;}
    .page-head h1 {font-size:34px;}
    .kpi-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
    .st-key-dash_row, .st-key-pantau_row {grid-template-columns:minmax(0,1fr);}
}
</style>
"""

LOGO = ('<div class="logo"><svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#2d6a4f" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/>'
        '<path d="M8 6V4h8v2"/><path d="M6 6l1 14h10l1-14"/><path d="M10 10v6M14 10v6"/></svg>'
        'TrashSpot</div>')


# ============================================================ helper umum
def H(s: str) -> str:
    """Rapatkan HTML agar tidak dianggap blok kode oleh markdown."""
    return "".join(line.strip() for line in s.splitlines())


esc = html.escape


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def fdate(ts, year: bool = False) -> str:
    base = f"{ts.day} {MONTHS[ts.month - 1]}"
    return f"{base} {ts.year}" if year else base


def pill(cat: str, cls: str = "", prefix: str = "") -> str:
    bg, fg, dot = CATS[cat]
    return (f'<span class="pill {cls}" style="background:{bg};color:{fg}">'
            f'<i style="background:{dot}"></i>{prefix}{cat}</span>')


def head(title: str, sub: str, right: str = "", extra_css: str = "") -> None:
    st.markdown(H(f'{extra_css}<div class="page-head"><div><h1>{title}</h1><p>{sub}</p></div>'
                  f'<div>{right}</div></div>'), unsafe_allow_html=True)


def shrink(img, max_w: int):
    h, w = img.shape[:2]
    return cv2.resize(img, (max_w, int(h * max_w / w))) if w > max_w else img


def img_b64(rel_path) -> str | None:
    if not isinstance(rel_path, str):
        return None
    p = BASE / rel_path
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else None


def go(page: str) -> None:
    st.session_state["page"] = page


def select_cam(no: int) -> None:
    st.session_state["sel_cam"] = no


def request_capture() -> None:
    st.session_state["capture_req"] = True


def go_history() -> None:
    st.session_state["hist_cam"] = st.session_state.get("sel_cam")
    st.session_state["page"] = "Riwayat"


@st.cache_resource(show_spinner="Memuat model YOLO26s...")
def get_detector(path: str, mtime: float) -> WasteDetector:
    return WasteDetector(path or None)


JS_CLICK = """
<script>
const d = window.parent.document;
if (!d.__tsBound) {
  d.__tsBound = true;
  d.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;               // klik langsung pada tombol: biarkan
    const card = e.target.closest('[class*="st-key-cam_"]');
    if (card) {                                            // klik kartu CCTV -> tombol pilih
      const m = card.className.match(/st-key-cam_(\\d+)/);
      const b = m && d.querySelector('.st-key-pick_' + m[1] + ' button');
      if (b) b.click();
      return;
    }
    const row = e.target.closest('[class*="st-key-hrow_"]');
    if (row && e.target.closest('img')) {                  // klik gambar riwayat -> perbesar
      const b = row.querySelector('button');
      if (b) b.click();
    }
  }, true);
}
</script>
"""


def inject_js() -> None:
    """Pasang klik cadangan (kartu CCTV & gambar riwayat) lewat iframe tersembunyi."""
    with st.container(key="js_host"):
        if hasattr(st, "iframe"):                      # Streamlit terbaru
            st.iframe(JS_CLICK, height=1)
        else:                                          # versi lama
            import streamlit.components.v1 as components
            components.html(JS_CLICK, height=0)


# ================================================================ top bar
def render_topbar() -> None:
    with st.container(key="topbar"):
        c1, c2, c3 = st.columns([1.4, 3, 1.6], vertical_alignment="center")
        c1.markdown(LOGO, unsafe_allow_html=True)
        with c2:
            for col, p in zip(st.columns([1, 1, 1], gap="small"), PAGES):
                col.button(p, key=f"nav_{p}", width="stretch", on_click=go, args=(p,),
                           type="primary" if st.session_state["page"] == p else "secondary")
        c3.markdown('<div class="loc-pill"><span>Desa Delod Peken · Tabanan</span></div>',
                    unsafe_allow_html=True)


# ============================================================== Dashboard
def page_dashboard(cams) -> None:
    cmap = {c.no: c for c in cams}
    latest = _latest_per_camera()
    last = _get_analyses(limit=1)
    stamp = (f"Pemeriksaan terakhir {fdate(last.iloc[0]['captured_at'], True)}"
             if not last.empty else "Belum ada pemeriksaan")
    head("Kondisi kebersihan desa", "Ringkasan pemeriksaan terakhir di setiap titik yang dipantau.",
         f'<span class="mono-pill">{stamp}</span>')
    if latest.empty:
        st.markdown(H('<div class="card"><h3>Belum ada data</h3><p>Buka menu <b>Pantau CCTV</b>. '
                      'Gambar akan diambil dan dianalisis otomatis, lalu ringkasannya muncul di sini.</p></div>'),
                    unsafe_allow_html=True)
        st.button("Buka Pantau CCTV", key="btn_check", type="primary", on_click=go, args=("Pantau CCTV",))
        return

    n_loc, total = len(latest), int(latest["total"].sum())
    agg: Counter = Counter()
    for c in latest["counts"]:
        agg.update(c)
    top_name = cap(pretty(agg.most_common(1)[0][0])) if agg else "—"
    n_high = int((latest["priority"] == "Tinggi").sum())

    def kpi(val, label, small=False, alert=False):
        return (f'<div class="kpi {"alert" if alert else ""}"><b class="{"sm" if small else ""}">{val}</b>'
                f'<span>{label}</span></div>')

    with st.container(key="kpi_row"):
        st.markdown(H('<div class="kpi-grid">'
                      + kpi(n_loc, "lokasi telah diperiksa")
                      + kpi(total, "sampah terdeteksi (pemeriksaan terakhir)")
                      + kpi(esc(top_name), "jenis paling banyak ditemukan", small=True)
                      + kpi(n_high, "lokasi prioritas tinggi", alert=True)
                      + '</div>'), unsafe_allow_html=True)

    # kiri: dua kartu bertumpuk (jarak pasti) | kanan: prioritas pembersihan
    lv = ""
    for cat in db.CATEGORIES:
        n = int((latest["category"] == cat).sum())
        lv += (f'<div class="lvl"><div class="lvl-top">{pill(cat)}<b>{n}</b></div>'
               f'<div class="bar"><span style="width:{n / n_loc * 100:.0f}%;background:{CATS[cat][2]}"></span></div></div>')
    top = agg.most_common(6)
    mx = max([n for _, n in top], default=1)
    tp = "".join(f'<div class="trow"><span>{esc(cap(pretty(k_)))}</span><div class="bar"><span '
                 f'style="width:{n / mx * 100:.0f}%;background:#2d6a4f"></span></div><b>{n}</b></div>'
                 for k_, n in top) or '<p class="muted-sm">Belum ada sampah terdeteksi.</p>'

    order = latest.assign(ci=latest["category"].map(db.CATEGORIES.index)).sort_values(
        ["ci", "total"], ascending=False)
    rows = ""
    for i, r in enumerate(order.itertuples(), 1):
        cam = cmap.get(r.camera_no)
        name = cam.label if cam else r.camera_label        # nama sama dengan yang dipakai di Riwayat
        banjar = cam.banjar if cam else ""
        rows += (f'<div class="prio-row"><span class="n">{i}</span><div><div class="nm">{esc(name)}</div>'
                 f'<div class="sb">{esc(banjar)} · {fdate(r.captured_at)}, {r.captured_at:%H:%M}</div></div>'
                 f'{pill(r.category)}<span class="it">{r.total} item</span></div>')

    with st.container(key="dash_row"):
        with st.container(key="dash_left"):
            with st.container(key="card_lvl"):
                st.markdown(H(f'<h3>Lokasi per tingkat kebersihan</h3>{lv}'), unsafe_allow_html=True)
            with st.container(key="card_jns"):
                st.markdown(H(f'<h3>Jenis sampah paling banyak</h3>{tp}'), unsafe_allow_html=True)
        with st.container(key="prio_card"):
            st.markdown(H(f'<h3>Prioritas pembersihan</h3><div class="muted-sm">Diurutkan dari yang paling '
                          f'mendesak.</div><div>{rows}</div>'), unsafe_allow_html=True)
            st.button("Periksa lokasi baru", key="btn_check", type="primary", on_click=go, args=("Pantau CCTV",))


# ============================================================== Riwayat
def page_history(cams) -> None:
    head("Riwayat pemeriksaan",
         "Pantau apakah suatu lokasi membaik, tetap, atau memburuk, termasuk setelah dibersihkan.")
    nos = [c.no for c in cams]
    cmap = {c.no: c for c in cams}
    if st.session_state.get("hist_cam") not in nos:
        st.session_state["hist_cam"] = nos[0]
    c1, c2 = st.columns([1, 2.3], gap="small", vertical_alignment="bottom")
    cam_no = c1.selectbox("Lokasi", nos, key="hist_cam",
                          format_func=lambda n: f"{cmap[n].label} · {cmap[n].banjar}")
    df = _get_analyses(camera_no=cam_no)
    if df.empty:
        st.markdown(H('<div class="card"><h3>Belum ada pemeriksaan</h3><p>Lokasi ini belum punya riwayat. '
                      'Buka <b>Pantau CCTV</b> agar gambar diambil dan dianalisis.</p></div>'),
                    unsafe_allow_html=True)
        return
    c2.markdown(f'<div class="pillrow">{pill(df.iloc[0]["category"], prefix="Kondisi terakhir: ")}</div>',
                unsafe_allow_html=True)

    hist = df.sort_values("captured_at").tail(14)
    mx = max(int(hist["total"].max()), 1)
    date_only = hist["captured_at"].dt.date.nunique() == len(hist)   # jam hanya jika ada tanggal kembar
    cols = ""
    for r in hist.itertuples():
        h = max(6, int(r.total / mx * 200))
        when = fdate(r.captured_at) if date_only else f"{fdate(r.captured_at)}<br>{r.captured_at:%H:%M}"
        extra = '<br><span>setelah dibersihkan</span>' if r.after_cleaning else ""
        cols += (f'<div class="col"><div class="plot"><b>{r.total}</b><div class="b" style="height:{h}px;'
                 f'background:{CATS[r.category][2]}"></div></div>'
                 f'<div class="xl">{when}{extra}</div></div>')
    with st.container(key="card_chart"):
        st.markdown(H(f'<h3>Jumlah sampah per pemeriksaan</h3><div class="chart">{cols}</div>'),
                    unsafe_allow_html=True)

    shown = st.session_state.get("hist_shown", 15)
    with st.container(key="hist_list"):
        st.markdown("<h3>Semua pemeriksaan</h3>", unsafe_allow_html=True)
        for r in df.head(shown).itertuples():
            with st.container(key=f"hrow_{r.id}"):
                c = st.columns([1.35, .75, 4.3, .55, 2.2, .95], gap="small", vertical_alignment="center")
                c[0].markdown(f'<span class="dt">{fdate(r.captured_at)} · {r.captured_at:%H:%M}</span>',
                              unsafe_allow_html=True)
                with c[1]:
                    with st.container(key=f"zc_{r.id}"):       # klik gambar = perbesar
                        thumb = BASE / r.thumb_path if isinstance(r.thumb_path, str) else None
                        if thumb is not None and thumb.exists():
                            st.image(str(thumb))
                        else:
                            st.markdown('<div class="ph"></div>', unsafe_allow_html=True)
                        if st.button("Perbesar", key=f"zb_{r.id}"):
                            show_detail(int(r.id))
                top = ", ".join(cap(pretty(k)) if i == 0 else pretty(k) for i, (k, _) in
                                enumerate(Counter(r.counts).most_common(3))) or "—"
                c[2].markdown(esc(top))
                c[3].markdown(f'<div class="r"><b>{r.total}</b></div>', unsafe_allow_html=True)
                c[4].markdown(f'<div class="r">{pill(r.category)}</div>', unsafe_allow_html=True)
                c[5].markdown(f'<div class="r">{r.priority}</div>', unsafe_allow_html=True)
        if len(df) > shown:
            st.button("Tampilkan lebih banyak", key="btn_more", on_click=more_rows)


def more_rows() -> None:
    st.session_state["hist_shown"] = st.session_state.get("hist_shown", 15) + 15


@st.dialog("Detail pemeriksaan", width="large")
def show_detail(analysis_id: int) -> None:
    r = _get_analysis(analysis_id)
    if r is None:
        st.warning("Data tidak ditemukan.")
        return
    photo = BASE / r["photo_path"] if r.get("photo_path") else None
    if photo is not None and photo.exists():
        st.image(str(photo), width="stretch")
    else:
        st.info("Gambar tidak tersedia.")
    st.markdown(pill(r["category"], "lg") + f'&nbsp; Prioritas <b>{r["priority"]}</b> · '
                f'<b>{r["total"]}</b> sampah terdeteksi', unsafe_allow_html=True)
    counts = r["counts"] or {}
    if counts:
        st.markdown(", ".join(f"{cap(pretty(k))}: **{n}**" for k, n in sorted(counts.items(), key=lambda x: -x[1])))
    ts = r["captured_at"]
    st.caption(f'{fdate(ts, True)} · {ts:%H:%M:%S} · {r["camera_label"]} · video {r.get("source_video") or "-"} · '
               f'{"manual" if r["trigger"] == "manual" else "otomatis"}'
               f'{" · setelah dibersihkan" if r["after_cleaning"] else ""} · model {r.get("model_version") or "-"}')


# ============================================================ Pantau CCTV
def card_info(cam, cat: str | None) -> str:
    foot = ('<div class="cam-off">Tidak ada sinyal</div>' if not cam.online
            else pill(cat, "wide") if cat else '<span class="pill wide muted">Menganalisis…</span>')
    return H(f'<div class="cam-title">{esc(cam.title)}</div><div class="cam-sub">{esc(cam.banjar)} · '
             f'{"online" if cam.online else "offline"}</div>{foot}')


def cond_html(cat: str | None, total: int | None) -> str:
    if cat is None:
        top, prio = '<span class="pill lg muted">Menganalisis…</span>', "—"
    else:
        top, prio = pill(cat, "lg"), db.PRIORITY[cat]
    return H(f'<div class="muted-sm">Kondisi kebersihan</div>{top}'
             f'<div class="kv"><span>Prioritas penanganan</span><b>{prio}</b></div>'
             f'<div class="kv"><span>Total sampah terdeteksi</span><b>{"—" if total is None else total}</b></div>')


def types_html(counts: Counter) -> str:
    rows = "".join(f'<div class="trow-c"><span>{esc(cap(k))}</span><b>{n}</b></div>'
                   for k, n in counts.most_common()) or '<p class="muted-sm">Tidak ada sampah terdeteksi.</p>'
    return H(f'<h3>Jenis dan jumlah</h3><div class="types-scroll">{rows}</div>')


def save_capture(cam, player_name: str, frame_idx: int, frame_s, dets, det, th, trigger: str,
                 after: bool) -> str:
    """Simpan gambar beranotasi + thumbnail + hasil analisis ke database."""
    now = datetime.now()
    drawn = draw_detections(frame_s, dets, cam.title,
                            f"{fdate(now, True)} · {now:%H:%M}")
    stem = f"cam{cam.no}_{now:%Y%m%d_%H%M%S}"
    photo, thumb = CAPTURE_DIR / f"{stem}.jpg", CAPTURE_DIR / f"{stem}_t.jpg"
    cv2.imwrite(str(photo), drawn, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(thumb), shrink(drawn, 160), [cv2.IMWRITE_JPEG_QUALITY, 80])
    counts = dict(Counter(pretty(d.class_name) for d in dets))
    cat = categorize(len(dets), th)
    db.add_analysis(camera_no=cam.no, camera_label=cam.label, captured_at=now,
                    source_video=player_name, frame_idx=frame_idx,
                    photo_path=photo.relative_to(BASE).as_posix(),
                    thumb_path=thumb.relative_to(BASE).as_posix(), total=len(dets),
                    category=cat, counts=counts, after_cleaning=after, trigger=trigger,
                    model_version=det.version)
    return cat


def page_cctv(cams, det: WasteDetector) -> None:
    ss = st.session_state
    th = db.get_thresholds()
    online = [c for c in cams if c.online]
    if not online:
        head("Pantau CCTV", "Gambar diambil otomatis dari kamera di tiap titik, lalu dianalisis oleh YOLO.")
        st.warning("Tidak ada video yang bisa dibaca. Isi path video di CAMERA_CONFIG "
                   "(bagian atas app.py), lalu simpan file.")
        return
    if ss.get("sel_cam") not in [c.no for c in online]:
        ss["sel_cam"] = online[0].no
    sel_cam = next(c for c in online if c.no == ss["sel_cam"])

    head("Pantau CCTV", "Gambar diambil otomatis dari kamera di tiap titik, lalu dianalisis oleh YOLO.",
         f'<span class="pill-online"><i></i>{len(online)} dari {len(cams)} kamera online</span>',
         extra_css=f"<style>.st-key-cam_{sel_cam.no}{{border-color:#2d6a4f !important;}}</style>")

    latest = {int(r.camera_no): r for r in _latest_per_camera().itertuples()}
    latest_cat = {no: r.category for no, r in latest.items()}

    # ---- 4 kartu kamera
    thumb_ph, info_ph = {}, {}
    for i in range(0, len(cams), 4):
        for col, cam in zip(st.columns(4, gap="small"), cams[i:i + 4]):
            with col.container(key=f"cam_{cam.no}"):
                thumb_ph[cam.no], info_ph[cam.no] = st.empty(), st.empty()
                thumb_ph[cam.no].markdown(f'<div class="thumb-ph {"" if cam.online else "hatch"}"></div>',
                                          unsafe_allow_html=True)
                info_ph[cam.no].markdown(card_info(cam, latest_cat.get(cam.no)), unsafe_allow_html=True)
                if cam.online:
                    st.button("Pilih kamera", key=f"pick_{cam.no}", on_click=select_cam, args=(cam.no,))

    # ---- video utama (kiri) + panel kanan: kondisi, jenis & jumlah, pengambilan otomatis
    with st.container(key="pantau_row"):
        with st.container(key="pv_left"):
            with st.container(key="main_card"):
                big_ph = st.empty()
                big_ph.markdown('<div class="main-skel">Memuat video…</div>', unsafe_allow_html=True)
            b1, b2, _ = st.columns([1.3, 1.15, 1.1])
            b1.button("Ambil gambar sekarang", key="btn_capture", type="primary", on_click=request_capture)
            b2.button("Lihat riwayat lokasi", key="btn_hist", on_click=go_history)
        with st.container(key="pv_right"):
            with st.container(key="card_cond"):
                cond_ph = st.empty()
            with st.container(key="card_types"):
                types_ph = st.empty()
            cond_ph.markdown(cond_html(latest_cat.get(sel_cam.no), None), unsafe_allow_html=True)
            types_ph.markdown(types_html(Counter()), unsafe_allow_html=True)
            with st.container(key="auto_card"):
                st.markdown("<h3>Pengambilan otomatis</h3>", unsafe_allow_html=True)
                label = st.selectbox("Ambil gambar setiap", list(INTERVALS), index=2, key="interval_label")
                st.caption("Setiap hasil analisis tersimpan otomatis ke riwayat lokasi. Ambang batas "
                           "kategori dikalibrasi bersama petugas desa.")

    # Loop pemutaran di bawah tidak pernah selesai, jadi Streamlit tidak sempat menghapus sisa
    # elemen halaman lain (mis. kartu "Semua pemeriksaan" dari Riwayat). Elemen lama itu ada di
    # posisi urutan setelah elemen terakhir halaman ini, jadi kita timpa dengan wadah kosong.
    for pad in ("pad_a", "pad_b", "pad_c"):
        with st.container(key=pad):
            st.empty()

    # ---- state pemutar
    players = ss.setdefault("players", {})
    for cam in online:
        p = players.get(cam.no)
        if p is None or p.videos != cam.videos:
            if p is not None:
                p.close()
            players[cam.no] = CameraPlayer(cam.videos)
    interval = INTERVALS[label]
    now = time.time()
    if ss.get("_applied_interval") != label or "next_due" not in ss:
        ss["_applied_interval"] = label
        ss["next_due"] = {
            cam.no: (float("inf") if interval is None else
                     max((latest[cam.no].captured_at.timestamp() if cam.no in latest else 0) + interval,
                         now + 3 * k))
            for k, cam in enumerate(online)}
    due = ss["next_due"]

    # ---- loop pemutaran (berhenti otomatis saat ada interaksi / ganti halaman)
    # Video selalu diputar penuh; YOLO jalan di thread terpisah supaya tidak menahan tampilan.
    smooth: deque = deque(maxlen=5)
    last_t: dict[int, float] = {}
    last_thumb: dict[int, float] = {}
    last_html: dict[str, str] = {}
    cur_dets: list = []
    have_dets = False
    fut = None
    last_infer = 0.0
    tick = 0
    while True:
        t0 = time.time()
        sel_pack = None
        for cam in online:
            is_sel = cam.no == sel_cam.no
            if not is_sel and t0 - last_thumb.get(cam.no, 0.0) < THUMB_EVERY:
                continue                                   # thumbnail tidak perlu secepat video utama
            p = players[cam.no]
            step = min(60, max(1, round((t0 - last_t.get(cam.no, t0 - 0.05)) * p.fps)))
            last_t[cam.no] = t0
            frame = p.next(step)
            if frame is None:
                continue
            if not is_sel:
                last_thumb[cam.no] = t0
                thumb_ph[cam.no].image(shrink(frame, 320), channels="BGR", output_format="JPEG", width="stretch")
                continue

            fs = shrink(frame, 960)
            if fut is not None and fut.done():             # ambil hasil deteksi yang sudah selesai
                try:
                    cur_dets, have_dets = fut.result(), True
                    smooth.append(len(cur_dets))
                except Exception:                          # noqa: BLE001
                    pass
                fut = None
            if fut is None and t0 - last_infer >= INFER_EVERY:
                fut = _EXEC.submit(det.predict, fs.copy())
                last_infer = t0
            sel_pack = (fs, cur_dets)
            now_dt = datetime.now()
            big_ph.image(draw_detections(fs, cur_dets, cam.title,
                                         f"{p.name} · {fdate(now_dt, True)} · {now_dt:%H:%M}"),
                         channels="BGR", output_format="JPEG", width="stretch")
            if have_dets:
                cat = categorize(int(np.median(smooth)), th)
                for key, ph, content in (("c", cond_ph, cond_html(cat, len(cur_dets))),
                                         ("t", types_ph, types_html(Counter(pretty(d.class_name)
                                                                            for d in cur_dets)))):
                    if last_html.get(key) != content:
                        ph.markdown(content, unsafe_allow_html=True)
                        last_html[key] = content

        # pengambilan manual (deteksi ulang tepat pada frame yang diambil)
        if ss.pop("capture_req", False) and sel_pack is not None:
            p = players[sel_cam.no]
            c = save_capture(sel_cam, p.name, p.idx, sel_pack[0], det.predict(sel_pack[0]), det, th,
                             "manual", bool(ss.get("mark_clean")))
            latest_cat[sel_cam.no] = c
            info_ph[sel_cam.no].markdown(card_info(sel_cam, c), unsafe_allow_html=True)
            st.toast("Gambar disimpan ke riwayat lokasi.", icon="✅")

        # pengambilan otomatis berkala
        tnow = time.time()
        for cam in online:
            if interval is None or tnow < due.get(cam.no, float("inf")):
                continue
            p = players[cam.no]
            if cam.no == sel_cam.no and sel_pack is not None and have_dets:
                fs, dets = sel_pack
            elif p.last is not None:
                fs = shrink(p.last, 960)
                dets = det.predict(fs)
            else:
                continue
            due[cam.no] = time.time() + interval      # dicatat dulu: kalau skrip dihentikan di tengah jalan,
            c = save_capture(cam, p.name, p.idx, fs, dets, det, th, "auto", False)  # tidak tersimpan dua kali
            latest_cat[cam.no] = c
            info_ph[cam.no].markdown(card_info(cam, c), unsafe_allow_html=True)

        tick += 1
        if MAX_TICKS and tick >= MAX_TICKS:
            break
        time.sleep(max(0.0, 0.04 - (time.time() - t0)))


def save_th() -> None:
    ss = st.session_state
    b, c, p = ss["th_b"], ss["th_c"], ss["th_p"]
    if b < c < p:
        db.set_thresholds(b, c, p)
    else:
        st.toast("Ambang harus berurutan: Bersih < Cukup < Perlu.", icon="⚠️")


# ================================================================== main
st.markdown(CSS, unsafe_allow_html=True)
st.session_state.setdefault("page", "Dashboard")

cams = build_cameras(CAMERA_CONFIG, BASE)
model_file = find_model(BASE)
detector = get_detector(str(model_file) if model_file else "",
                        model_file.stat().st_mtime if model_file else 0.0)

render_topbar()
inject_js()
if detector.demo:
    st.warning(f"{detector.error} Deteksi saat ini hanya simulasi.")

# Halaman dibungkus satu slot: saat berpindah halaman, seluruh isi halaman lama langsung
# diganti (loop Pantau CCTV tidak pernah "selesai", jadi tanpa ini kartu halaman lama tertinggal).
page = st.session_state["page"]
with st.empty().container(key="page"):
    if page == "Dashboard":
        page_dashboard(cams)
    elif page == "Riwayat":
        page_history(cams)
    else:
        page_cctv(cams, detector)