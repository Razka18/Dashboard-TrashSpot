"""Konfigurasi kamera (path video diisi manual) dan pemutar playlist video per kamera."""
from __future__ import annotations

import os

# Matikan multi-thread decoder FFmpeg SEBELUM cv2 diimpor. Ini mencegah error
# "Assertion fctx->async_lock failed at libavcodec/pthread_frame.c".
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")

import re
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def _natural(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


@dataclass
class Camera:
    no: int
    label: str
    banjar: str
    videos: list[Path]

    @property
    def online(self) -> bool:
        return bool(self.videos)

    @property
    def title(self) -> str:
        return f"CCTV {self.no} · {self.label}"


def _expand(entry, base: Path) -> list[Path]:
    """Satu entri path -> daftar file video. Boleh berupa file ATAU folder."""
    p = Path(str(entry)).expanduser()
    if not p.is_absolute():
        p = base / p
    if p.is_dir():
        return sorted((f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT), key=_natural)
    if p.is_file() and p.suffix.lower() in VIDEO_EXT:
        return [p]
    print(f"[TrashSpot] PERINGATAN: path video tidak ditemukan / bukan video: {p}")
    return []


def build_cameras(config: dict, base: Path, slots: int = 4) -> list[Camera]:
    """Bangun daftar kamera dari CAMERA_CONFIG di app.py. Slot tanpa video = offline."""
    cams = []
    for no in sorted(set(range(1, slots + 1)) | set(config)):
        c = config.get(no, {})
        entries = c.get("videos", [])
        if isinstance(entries, (str, Path)):
            entries = [entries]
        videos: list[Path] = []
        for e in entries:
            videos += _expand(e, base)
        cams.append(Camera(no, c.get("label") or f"Titik {no}",
                           c.get("banjar") or "[Nama banjar]", videos))
    return cams


class CameraPlayer:
    """Memutar semua video milik satu kamera secara berurutan (berulang).

    Semua akses ke cv2.VideoCapture dikunci (lock) karena Streamlit dapat menjalankan
    skrip lama dan baru bersamaan saat rerun; akses bersamaan ke FFmpeg itulah
    penyebab crash `async_lock`.
    """

    def __init__(self, videos: list[Path]):
        self.videos = list(videos)
        self.vi = 0
        self.cap = None
        self.fps = 25.0
        self.idx = 0
        self.last = None
        self.name = ""
        self._lock = threading.RLock()
        if self.videos:
            with self._lock:
                self._open()

    def _open(self) -> None:  # dipanggil saat lock dipegang
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        path = self.videos[self.vi]
        self.cap = cv2.VideoCapture(str(path))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.fps = fps if 0 < fps < 120 else 25.0
        self.name = path.name
        self.idx = 0

    def close(self) -> None:
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def next(self, step: int = 1):
        """Maju `step` frame (agar kecepatan putar mengikuti waktu nyata)."""
        with self._lock:
            if not self.videos:
                return None
            got, fails = 0, 0
            while got < max(1, step):
                if self.cap is not None and self.cap.grab():
                    got += 1
                    fails = 0
                    self.idx += 1
                    continue
                fails += 1
                if fails > len(self.videos) + 1:   # semua video tak bisa dibaca
                    break
                self.vi = (self.vi + 1) % len(self.videos)
                self._open()
            if got and self.cap is not None:
                ok, frame = self.cap.retrieve()
                if ok and frame is not None:
                    self.last = frame
            return self.last