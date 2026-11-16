"""detector.py - deteksi + tracking sampah dari video CCTV (YOLO + ByteTrack).

Pemakaian:
    from detector import analyze_video
    hasil = analyze_video("video.mp4", "best.pt", out_dir="output")
    hasil["summary"], hasil["tracks"], hasil["video"], hasil["csv"]
"""
import csv
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PALETTE = [(0, 170, 255), (80, 220, 80), (255, 140, 40), (230, 80, 200), (80, 200, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Tingkat kebersihan sesuai dashboard (urut dari terbersih ke terkotor)
STATUS = {
    "bersih":                   {"label": "Bersih",                   "hex": "#1F7A5C", "bgr": (92, 122, 31),  "prioritas": "Rendah"},
    "cukup_bersih":             {"label": "Cukup Bersih",             "hex": "#8DA52E", "bgr": (46, 165, 141), "prioritas": "Rendah"},
    "perlu_dibersihkan":        {"label": "Perlu Dibersihkan",        "hex": "#E0A21B", "bgr": (27, 162, 224), "prioritas": "Sedang"},
    "sangat_perlu_dibersihkan": {"label": "Sangat Perlu Dibersihkan", "hex": "#C2410C", "bgr": (12, 65, 194),  "prioritas": "Tinggi"},
}
STATUS_URUT = list(STATUS)

LABEL_JENIS = {"botol_plastik": "Botol plastik", "kantong_plastik": "Kantong plastik",
               "kaleng": "Kaleng", "kardus": "Kardus"}


def label_jenis(nama):
    return LABEL_JENIS.get(nama, nama.replace("_", " ").capitalize())


def klasifikasi_kebersihan(jumlah, batas=(5, 15, 28)):
    """Jumlah sampah -> status kebersihan. batas = (maks Bersih, maks Cukup Bersih, maks Perlu Dibersihkan)."""
    b1, b2, b3 = batas
    if jumlah <= b1:
        kode, info = "bersih", f"Lokasi bersih. Hanya {jumlah} sampah terdeteksi."
    elif jumlah <= b2:
        kode, info = "cukup_bersih", f"Lokasi cukup bersih. Ada {jumlah} sampah, masih aman tetapi perlu dipantau."
    elif jumlah <= b3:
        kode, info = "perlu_dibersihkan", f"Lokasi perlu dibersihkan. Ada {jumlah} sampah, jadwalkan pembersihan."
    else:
        kode, info = "sangat_perlu_dibersihkan", f"Lokasi sangat perlu dibersihkan. Ada {jumlah} sampah, segera tangani."
    s = STATUS[kode]
    return {"kode": kode, "label": s["label"], "prioritas": s["prioritas"], "jumlah_sampah": int(jumlah),
            "keterangan": info, "warna_hex": s["hex"], "tingkat": STATUS_URUT.index(kode), "batas": list(batas)}


def analyze_video(video_path, model, out_dir="output", tag=None, conf=0.25, imgsz=640,
                  tracker="bytetrack.yaml", min_hits=5, static_sec=2.0, static_move=0.06,
                  out_width=1280, device=None, save_video=True, progress=None,
                  hitung_dari="menetap", batas=(5, 15, 28), compensate_camera=True):
    """Deteksi + tracking satu video.

    model         : path .pt atau objek YOLO
    min_hits      : track dihitung valid kalau muncul >= N frame (buang flicker)
    static_sec    : dianggap menetap kalau terlihat >= N detik ...
    static_move   : ... dan bergeser < fraksi ini dari diagonal frame
    save_video    : False = hanya hitung (lebih cepat), tanpa video hasil
    progress      : callback progress(frame_ke, total_frame), mis. untuk st.progress
    hitung_dari   : "menetap" (sampah diam saja) atau "unik" (semua sampah) -> dasar penentuan status
    compensate_camera : True = gerak kamera (goyang/pan) dibuang dulu sebelum menilai apakah sampah diam
    batas         : (maks Bersih, maks Cukup Bersih, maks Perlu Dibersihkan); di atasnya = Sangat Perlu Dibersihkan

    Return dict: video, csv, json, fps, frames, duration_sec, summary (ada "status" & "jumlah_per_jenis"), tracks
    """
    video_path = str(video_path)
    tag = tag or Path(video_path).stem
    os.makedirs(out_dir, exist_ok=True)
    mdl = model if isinstance(model, YOLO) else YOLO(str(model))
    names = mdl.names

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video tidak bisa dibuka: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    sc = min(1.0, out_width / W)
    ow, oh = int(W * sc) // 2 * 2, int(H * sc) // 2 * 2
    diag = float(np.hypot(W, H))
    fs = max(0.45, ow / 1280 * 0.6)
    th = max(1, round(ow / 640))

    raw_out = os.path.join(out_dir, f"{tag}_raw.mp4")
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh)) if save_video else None

    tracks = {}

    def extent(t):
        """Sebaran posisi track (4 x simpangan baku, gerak kamera sudah dibuang) dibagi diagonal frame."""
        n = t["n"]
        vx = max(0.0, t["sxx"] / n - (t["sx"] / n) ** 2)
        vy = max(0.0, t["syy"] / n - (t["sy"] / n) ** 2)
        return 4.0 * float(np.sqrt(vx + vy)) / diag

    def is_static(t):
        span = (t["last"] - t["first"] + 1) / fps
        return span >= static_sec and extent(t) <= static_move

    def tally():
        uniq, stat = Counter(), Counter()
        for t in tracks.values():
            if t["n"] >= min_hits:
                c = t["votes"].most_common(1)[0][0]
                uniq[c] += 1
                stat[c] += int(is_static(t))
        return uniq, stat

    # estimasi gerak kamera antar frame (translasi global) -> posisi sampah dihitung di "koordinat dunia"
    gw = 320
    gh = max(2, int(round(gw * H / W)))
    hann = cv2.createHanningWindow((gw, gh), cv2.CV_32F)
    prev_g, prev_textured, cam_x, cam_y = None, False, 0.0, 0.0

    fi = -1
    for fi, r in enumerate(mdl.track(source=video_path, stream=True, conf=conf, imgsz=imgsz,
                                     device=device, tracker=tracker, verbose=False)):
        if compensate_camera:
            g = cv2.cvtColor(cv2.resize(r.orig_img, (gw, gh), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.float32)
            textured = float(g.std()) >= 3.0           # frame polos/gelap -> phase correlation memberi geseran palsu
            if prev_g is not None and textured and prev_textured:
                (dx, dy), resp = cv2.phaseCorrelate(prev_g, g, hann)
                if resp > 0.1 and abs(dx) < gw * 0.25 and abs(dy) < gh * 0.25:
                    cam_x -= dx * W / gw
                    cam_y -= dy * H / gh
            prev_g, prev_textured = g, textured

        frame = cv2.resize(r.orig_img, (ow, oh), interpolation=cv2.INTER_AREA) if save_video else None

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            cls = r.boxes.cls.int().cpu().tolist()
            cfs = r.boxes.conf.cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()
            for tid, c, cf, b in zip(ids, cls, cfs, xyxy):
                cx, cy = (b[0] + b[2]) / 2 - cam_x, (b[1] + b[3]) / 2 - cam_y
                t = tracks.get(tid)
                if t is None:
                    t = tracks[tid] = dict(first=fi, last=fi, n=0, votes=Counter(), confsum=0.0,
                                           sx=0.0, sy=0.0, sxx=0.0, syy=0.0)
                t["last"] = fi; t["n"] += 1; t["votes"][c] += 1; t["confsum"] += cf
                t["sx"] += cx; t["sy"] += cy; t["sxx"] += cx * cx; t["syy"] += cy * cy

                if frame is None or t["n"] < min_hits:
                    continue
                static = is_static(t)
                col = (0, 0, 255) if static else PALETTE[c % len(PALETTE)]
                x1, y1, x2, y2 = [int(v * sc) for v in b]
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, th + (1 if static else 0))
                label = f"#{tid} {names[c]} {cf:.2f}" + (" MENETAP" if static else "")
                (tw, thh), bl = cv2.getTextSize(label, FONT, fs, 1)
                ty = max(y1, thh + bl + 2)
                cv2.rectangle(frame, (x1, ty - thh - bl - 2), (x1 + tw + 4, ty + 2), col, -1)
                cv2.putText(frame, label, (x1 + 2, ty - bl), FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)

        if frame is not None:
            uniq, stat = tally()
            n_basis = sum(stat.values()) if hitung_dari == "menetap" else sum(uniq.values())
            st = klasifikasi_kebersihan(n_basis, batas)
            rows = [f"STATUS: {st['label'].upper()}",
                    f"Sampah unik   : {sum(uniq.values())}", f"Sampah menetap: {sum(stat.values())}"]
            rows += [f"  {label_jenis(names[i])}: {uniq[i]} ({stat[i]} menetap)" for i in sorted(names)]
            lh = int(26 * fs / 0.6 * 0.6) + 8
            pw = int(max(cv2.getTextSize(s, FONT, fs, 1)[0][0] for s in rows)) + 20
            ph = lh * len(rows) + 12
            overlay = frame.copy()
            cv2.rectangle(overlay, (8, 8), (8 + pw, 8 + ph), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
            cv2.rectangle(frame, (8, 8), (8 + pw, 8 + lh + 6), STATUS[st["kode"]]["bgr"], -1)   # bar status berwarna
            for k, s in enumerate(rows):
                cv2.putText(frame, s, (18, 8 + 6 + lh * (k + 1) - 6), FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)

        if progress and (fi + 1) % 10 == 0:
            progress(fi + 1, total)
    if writer is not None:
        writer.release()
    if progress:
        progress(total, total)

    final = None
    if save_video:
        final = os.path.join(out_dir, f"{tag}_hasil.mp4")
        if shutil.which("ffmpeg"):          # H.264 supaya bisa diputar di browser / Streamlit
            subprocess.run(["ffmpeg", "-y", "-i", raw_out, "-c:v", "libx264", "-preset", "fast",
                            "-crf", "26", "-pix_fmt", "yuv420p", "-an", final],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(raw_out)
        else:
            os.replace(raw_out, final)

    uniq, stat = tally()
    per_kelas = {names[i]: {"unik": int(uniq[i]), "menetap": int(stat[i])} for i in sorted(names)}
    counts = stat if hitung_dari == "menetap" else uniq
    status = klasifikasi_kebersihan(int(sum(counts.values())), batas)
    status["dihitung_dari"] = hitung_dari
    summary = {"total_unik": int(sum(uniq.values())), "total_menetap": int(sum(stat.values())),
               "per_kelas": per_kelas,
               "jumlah_per_jenis": {label_jenis(names[i]): int(counts[i]) for i in sorted(names)},
               "status": status}

    rows = []
    for tid, t in sorted(tracks.items()):
        if t["n"] < min_hits:
            continue
        rows.append({
            "track_id": int(tid),
            "kelas": names[t["votes"].most_common(1)[0][0]],
            "frame_awal": int(t["first"]),
            "frame_akhir": int(t["last"]),
            "muncul_dtk": round(t["first"] / fps, 2),
            "hilang_dtk": round(t["last"] / fps, 2),
            "durasi_dtk": round((t["last"] - t["first"] + 1) / fps, 2),
            "bergeser_pct": round(extent(t) * 100, 2),
            "conf_rata2": round(t["confsum"] / t["n"], 3),
            "menetap": int(is_static(t)),
        })

    csv_path = os.path.join(out_dir, f"{tag}_tracking.csv")
    cols = ["track_id", "kelas", "frame_awal", "frame_akhir", "muncul_dtk", "hilang_dtk",
            "durasi_dtk", "bergeser_pct", "conf_rata2", "menetap"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    json_path = os.path.join(out_dir, f"{tag}_ringkasan.json")
    hasil = {"video": final, "csv": csv_path, "json": json_path, "fps": float(fps), "frames": fi + 1,
             "duration_sec": round((fi + 1) / fps, 2), "summary": summary, "tracks": rows}
    with open(json_path, "w") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False)
    return hasil


def print_summary(hasil):
    s = hasil["summary"]
    print(f"{'Kelas':<18}{'Unik':>6}{'Menetap':>10}")
    for k, v in s["per_kelas"].items():
        print(f"{k:<18}{v['unik']:>6}{v['menetap']:>10}")
    print(f"{'TOTAL':<18}{s['total_unik']:>6}{s['total_menetap']:>10}")
    st = s["status"]
    print(f"\nKONDISI KEBERSIHAN: {st['label'].upper()} | prioritas {st['prioritas']} "
          f"| {st['jumlah_sampah']} sampah ({st['dihitung_dari']})")
    print(st["keterangan"])
