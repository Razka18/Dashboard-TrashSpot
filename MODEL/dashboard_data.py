"""dashboard_data.py - riwayat pemeriksaan + ringkasan untuk dashboard TrashSpot.

Alur:
    hasil = analyze_video(...)                                   # detector.py
    catat_pemeriksaan(hasil, lokasi="Taman", banjar="Banjar X", kamera="CCTV 1", path="riwayat.json")
    dash = ringkasan_dashboard(muat_riwayat("riwayat.json"))     # halaman Dashboard
    rwy  = riwayat_lokasi(muat_riwayat("riwayat.json"), "Taman") # halaman Riwayat
    kartu_html(hasil, lokasi="Taman", kamera="CCTV 1")           # panel kanan halaman Pantau CCTV
"""
import json
import os
from collections import Counter
from datetime import datetime

import cv2

from detector import STATUS, STATUS_URUT, klasifikasi_kebersihan, label_jenis

FORMAT_WAKTU = "%Y-%m-%d %H:%M"


# ---------------------------------------------------------------- penyimpanan
def muat_riwayat(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _simpan(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def simpan_thumbnail(video_path, out_path, posisi=0.5, lebar=320):
    """Ambil 1 frame dari video (default tengah video) sebagai thumbnail riwayat."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = False, None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * posisi))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (lebar, int(h * lebar / w)))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, frame)
    return out_path


def _record(rid, lokasi, banjar, kamera, waktu, per_jenis, status, thumbnail=None):
    per_jenis = {k: int(v) for k, v in per_jenis.items()}
    urut = sorted(((k, v) for k, v in per_jenis.items() if v > 0), key=lambda x: -x[1])
    return {
        "id": rid, "lokasi": lokasi, "banjar": banjar, "kamera": kamera, "waktu": waktu,
        "jumlah_sampah": int(sum(per_jenis.values())),
        "per_jenis": per_jenis,
        "jenis_teks": ", ".join(k for k, _ in urut) or "-",
        "status": {k: status[k] for k in ("kode", "label", "prioritas", "warna_hex", "keterangan")},
        "thumbnail": thumbnail,
    }


def catat_pemeriksaan(hasil, lokasi, banjar="", kamera="", waktu=None, path="riwayat.json"):
    """Simpan hasil analyze_video() sebagai 1 baris riwayat pemeriksaan. Return record-nya."""
    s = hasil["summary"]
    data = muat_riwayat(path)
    rid = max([r["id"] for r in data], default=0) + 1
    waktu = waktu or datetime.now().strftime(FORMAT_WAKTU)
    thumb = None
    if hasil.get("video") and os.path.exists(hasil["video"]):
        thumb = simpan_thumbnail(hasil["video"], os.path.join(
            os.path.dirname(os.path.abspath(path)), "thumb", f"pemeriksaan_{rid}.jpg"))
    rec = _record(rid, lokasi, banjar, kamera, waktu, s["jumlah_per_jenis"], s["status"], thumb)
    data.append(rec)
    _simpan(path, data)
    return rec


# ---------------------------------------------------------------- halaman Dashboard
def terbaru_per_lokasi(riwayat):
    """Pemeriksaan paling baru dari tiap lokasi."""
    t = {}
    for r in sorted(riwayat, key=lambda r: r["waktu"]):
        t[r["lokasi"]] = r
    return list(t.values())


def ringkasan_dashboard(riwayat):
    """Data halaman 'Kondisi kebersihan desa' (kartu angka, lokasi per tingkat, prioritas, jenis terbanyak)."""
    terbaru = terbaru_per_lokasi(riwayat)
    jenis = Counter()
    for r in terbaru:
        jenis.update(r["per_jenis"])

    per_tingkat = []
    for kode in STATUS_URUT:
        per_tingkat.append({"kode": kode, "label": STATUS[kode]["label"], "warna_hex": STATUS[kode]["hex"],
                            "jumlah_lokasi": sum(1 for r in terbaru if r["status"]["kode"] == kode)})

    urut = sorted(terbaru, key=lambda r: (-STATUS_URUT.index(r["status"]["kode"]), -r["jumlah_sampah"]))
    prioritas = [{"urutan": i, "lokasi": r["lokasi"], "banjar": r["banjar"], "status": r["status"]["label"],
                  "kode": r["status"]["kode"], "warna_hex": r["status"]["warna_hex"],
                  "jumlah_sampah": r["jumlah_sampah"]} for i, r in enumerate(urut, 1)]

    return {
        "lokasi_dipantau": len(terbaru),
        "total_sampah": int(sum(r["jumlah_sampah"] for r in terbaru)),
        "jenis_terbanyak": jenis.most_common(1)[0][0] if jenis else "-",
        "lokasi_prioritas_tinggi": sum(1 for r in terbaru if r["status"]["kode"] == "sangat_perlu_dibersihkan"),
        "pembaruan_terakhir": max((r["waktu"] for r in terbaru), default="-"),
        "lokasi_per_tingkat": per_tingkat,
        "jenis_sampah_terbanyak": [{"jenis": k, "jumlah": int(v)} for k, v in jenis.most_common()],
        "prioritas_pembersihan": prioritas,
    }


# ---------------------------------------------------------------- halaman Riwayat
def daftar_lokasi(riwayat):
    return sorted({r["lokasi"] for r in riwayat})


def riwayat_lokasi(riwayat, lokasi):
    """Data halaman 'Riwayat pemeriksaan' untuk satu lokasi: grafik batang + daftar semua pemeriksaan."""
    rows = sorted((r for r in riwayat if r["lokasi"] == lokasi), key=lambda r: r["waktu"])
    return {
        "lokasi": lokasi,
        "banjar": rows[-1]["banjar"] if rows else "",
        "kondisi_terbaru": rows[-1]["status"] if rows else None,
        "grafik": [{"waktu": r["waktu"], "jumlah_sampah": r["jumlah_sampah"],
                    "status": r["status"]["label"], "warna_hex": r["status"]["warna_hex"]} for r in rows],
        "daftar": [{"waktu": r["waktu"], "thumbnail": r["thumbnail"], "jenis": r["jenis_teks"],
                    "jumlah_sampah": r["jumlah_sampah"], "status": r["status"]["label"],
                    "warna_hex": r["status"]["warna_hex"], "prioritas": r["status"]["prioritas"]}
                   for r in reversed(rows)],
    }


# ---------------------------------------------------------------- halaman Pantau CCTV
def kartu_html(hasil, lokasi="", kamera=""):
    """Panel kanan 'Pantau CCTV': kondisi kebersihan, prioritas penanganan, total sampah, jenis dan jumlah."""
    s = hasil["summary"]
    st = s["status"]
    baris = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #e3e7dc">'
        f'<span>{k}</span><b>{v}</b></div>' for k, v in s["jumlah_per_jenis"].items())
    return f"""
<div style="font-family:sans-serif;max-width:360px;background:#f1f4ee;padding:14px;border-radius:14px">
  <div style="font-size:12px;color:#666">{kamera} {('· ' + lokasi) if lokasi else ''}</div>
  <div style="background:#fff;border-radius:12px;padding:14px;margin:8px 0">
    <div style="font-size:12px;color:#666">Kondisi kebersihan</div>
    <div style="display:inline-block;margin:6px 0;padding:6px 14px;border-radius:999px;font-weight:700;
         color:{st['warna_hex']};background:{st['warna_hex']}22">● {st['label']}</div>
    <div style="display:flex;justify-content:space-between;font-size:13px"><span>Prioritas penanganan</span><b>{st['prioritas']}</b></div>
    <div style="display:flex;justify-content:space-between;font-size:13px"><span>Total sampah terdeteksi</span><b>{st['jumlah_sampah']}</b></div>
    <div style="font-size:12px;color:#666;margin-top:6px">{st['keterangan']}</div>
  </div>
  <div style="background:#fff;border-radius:12px;padding:14px">
    <div style="font-weight:700;margin-bottom:4px">Jenis dan jumlah</div>{baris}
  </div>
</div>"""


# ---------------------------------------------------------------- data contoh (untuk uji dashboard tanpa video)
def data_contoh(path, batas=(5, 15, 28)):
    """Isi riwayat.json dengan data tiruan mirip mockup, supaya tim dashboard bisa langsung mencoba."""
    def bagi(n):
        a, b, c = round(n * .40), round(n * .30), round(n * .15)
        return {"Botol plastik": a, "Kantong plastik": b, "Kaleng": c, "Kardus": max(0, n - a - b - c)}

    banjar = "Banjar Contoh"
    contoh = [
        ("Sungai", "CCTV 3", "2026-10-12 07:30", 41), ("Tempat wisata", "CCTV 4", "2026-10-12 08:00", 33),
        ("Jalan desa", "CCTV 1", "2026-10-12 08:30", 19), ("Pantai", "CCTV 5", "2026-10-12 09:00", 3),
        ("Balai banjar", "CCTV 6", "2026-10-12 09:30", 10), ("Pasar", "CCTV 7", "2026-10-12 10:00", 14),
        ("Lapangan", "CCTV 8", "2026-10-12 10:30", 6),
        ("Taman", "CCTV 2", "2026-09-14 08:00", 12), ("Taman", "CCTV 2", "2026-09-21 08:00", 24),
        ("Taman", "CCTV 2", "2026-09-26 08:00", 36), ("Taman", "CCTV 2", "2026-10-05 08:00", 4),
        ("Taman", "CCTV 2", "2026-10-12 08:00", 8),
    ]
    data = []
    for i, (lok, cam, waktu, n) in enumerate(contoh, 1):
        data.append(_record(i, lok, banjar, cam, waktu, bagi(n), klasifikasi_kebersihan(n, batas)))
    _simpan(path, data)
    return data
