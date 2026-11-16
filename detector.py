"""Detektor sampah YOLO26s untuk TrashSpot.

Model otomatis dicari di folder MODEL/ (mis. "MODEL/best (2).pt") lalu langsung
dimuat. Jika model tidak ditemukan / gagal dimuat, detektor jatuh ke mode simulasi
agar antarmuka tetap bisa dibuka (ditandai `demo=True`).
"""
from __future__ import annotations

import os
import random
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEMO_CLASSES = ["botol plastik", "kantong plastik", "kaleng", "kardus"]
LIME = (106, 245, 200)      # BGR dari #c8f56a
INK = (34, 43, 28)          # BGR dari #1c2b22


@dataclass
class Det:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def pretty(name: str) -> str:
    """botol_plastik -> 'botol plastik'."""
    return str(name).replace("_", " ").replace("-", " ").strip().lower()


def find_model(base: Path) -> Path | None:
    """Cari file .pt: env MODEL_PATH, lalu MODEL/, model/, models/, lalu root."""
    env = os.environ.get("MODEL_PATH")
    if env and Path(env).exists():
        return Path(env)
    cands: list[Path] = []
    for folder in (base / "MODEL", base / "model", base / "models", base):
        if folder.is_dir():
            cands += list(folder.glob("*.pt"))
    if not cands:
        return None
    # utamakan nama berawalan "best", lalu yang paling baru
    cands.sort(key=lambda p: (not p.name.lower().startswith("best"), -p.stat().st_mtime))
    return cands[0]


class WasteDetector:
    def __init__(self, model_path: str | None, conf: float = 0.25, iou: float = 0.45,
                 imgsz: int = 640):
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.model = None
        self._lock = threading.Lock()
        self.error: str | None = None
        self.names: list[str] = list(DEMO_CLASSES)
        self.version = "simulasi"
        if not model_path:
            self.error = "File model (.pt) tidak ditemukan di folder MODEL."
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.names = [self.model.names[i] for i in sorted(self.model.names)]
            self.version = Path(model_path).name
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.error = f"Gagal memuat model: {exc}"

    @property
    def demo(self) -> bool:
        return self.model is None

    def predict(self, frame: np.ndarray) -> list[Det]:
        if self.model is None:
            return self._simulate(frame)
        with self._lock:                      # satu inferensi dalam satu waktu
            res = self.model.predict(frame, conf=self.conf, iou=self.iou,
                                     imgsz=self.imgsz, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        return [Det(self.names[k] if k < len(self.names) else str(k), float(c),
                    float(x1), float(y1), float(x2), float(y2))
                for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls)]

    def _simulate(self, frame: np.ndarray) -> list[Det]:
        rng = random.Random(zlib.crc32(np.ascontiguousarray(frame[::64, ::64]).tobytes()))
        h, w = frame.shape[:2]
        out = []
        for _ in range(rng.choice([1, 2, 3, 4, 6, 9])):
            bw = rng.randint(int(w * .06), int(w * .16))
            bh = int(bw * rng.uniform(.8, 1.4))
            x, y = rng.randint(0, max(1, w - bw)), rng.randint(0, max(1, h - bh))
            out.append(Det(rng.choice(self.names), rng.uniform(.4, .95), x, y, x + bw, y + bh))
        return out


def _tag(img, text: str, x: int, y: int, scale: float, anchor_right: bool = False) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad = int(8 * scale / .5)
    x0 = x - tw - 2 * pad if anchor_right else x
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y), (x0 + tw + 2 * pad, y + th + base + pad), (38, 42, 36), -1)
    cv2.addWeighted(overlay, .8, img, .2, 0, img)
    cv2.putText(img, text, (x0 + pad, y + th + pad // 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, dets: list[Det], left_tag: str = "",
                    right_tag: str = "") -> np.ndarray:
    """Kotak hijau-limau + label gaya TrashSpot, plus tag kamera & waktu."""
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(.45, w / 1400)
    thick = max(2, w // 400)
    for d in dets:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, LIME, thick)
        label = f"{pretty(d.class_name)} {d.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        y0 = max(p1[1] - th - base - 6, 0)
        cv2.rectangle(out, (p1[0], y0), (p1[0] + tw + 10, y0 + th + base + 6), LIME, -1)
        cv2.putText(out, label, (p1[0] + 5, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, INK, 1, cv2.LINE_AA)
    m = int(w * .02)
    if left_tag:
        _tag(out, left_tag, m, m, scale)
    if right_tag:
        _tag(out, right_tag, w - m, m, scale, anchor_right=True)
    return out