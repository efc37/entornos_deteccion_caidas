"""Extracción de esqueletos con YOLOv8-Pose (privacidad por diseño).

Cada fotograma se procesa en memoria y se descarta inmediatamente: en disco solo
se guardan las coordenadas de los 17 puntos clave (COCO) de la persona seguida,
junto con la etiqueta por frame. Nunca se almacenan imágenes.

Salida: data/skeletons/<dataset>/<secuencia>.npz con
    kps      (T, 17, 3) float32  x, y (píxeles) y confianza; 0 si no hay detección
    box      (T, 4)     float32  bbox de la persona (NaN si no hay detección)
    labels   (T,)       int8     0 normal, 1 cayendo, 2 en el suelo tras caer
    fps, width, height, is_fall, group, dataset, seq_id
    pose_ms  tiempo medio de inferencia de pose por frame (coste computacional)
    jpeg_bytes tamaño medio de un fotograma en JPEG (solo para comparar el coste
               de almacenamiento frente al esqueleto; la imagen no se guarda)

Uso:
    python -m src.extract_skeletons --dataset urfd le2i [--device 0] [--model yolov8n-pose.pt]
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .datasets import IMG_EXT, Sequence, index_dataset


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def iter_frames(source: Path, half: str = "auto"):
    """Itera los fotogramas de un vídeo o de una carpeta de imágenes (BGR)."""
    if source.is_dir():
        for p in sorted((q for q in source.iterdir() if q.suffix.lower() in IMG_EXT), key=_natural_key):
            img = cv2.imread(str(p))
            if img is not None:
                yield img
        return
    cap = cv2.VideoCapture(str(source))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            # Los mp4 de URFD pueden mostrar profundidad y RGB lado a lado
            if half != "none" and w >= 1.9 * h:
                frame = frame[:, w // 2:] if half in ("auto", "right") else frame[:, : w // 2]
            yield frame
    finally:
        cap.release()


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def select_person(boxes: np.ndarray, confs: np.ndarray, prev_box: np.ndarray | None) -> int:
    """Seguimiento simple de una persona: mayor IoU con la anterior o, si no,
    la detección más grande y fiable."""
    if prev_box is not None and not np.isnan(prev_box).any():
        ious = np.array([_iou(b, prev_box) for b in boxes])
        if ious.max() > 0.2:
            return int(ious.argmax())
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas * confs))


class PoseExtractor:
    def __init__(self, model_name: str = C.POSE_MODEL, device: str | None = None):
        from ultralytics import YOLO  # import diferido: el resto del proyecto no lo necesita
        self.model = YOLO(model_name)
        self.device = device
        self.prev_box = None

    def reset(self):
        self.prev_box = None

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (kps (17,3), box (4,)) de la persona seguida en un fotograma."""
        res = self.model(frame, verbose=False, conf=C.POSE_CONF, device=self.device)[0]
        kps = np.zeros((C.N_KP, 3), np.float32)
        box = np.full(4, np.nan, np.float32)
        if res.boxes is not None and len(res.boxes) and res.keypoints is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            i = select_person(boxes, confs, self.prev_box)
            kps = res.keypoints.data[i].cpu().numpy().astype(np.float32)
            if kps.shape[1] == 2:  # algunas versiones no devuelven confianza
                kps = np.concatenate([kps, np.ones((C.N_KP, 1), np.float32)], 1)
            box = boxes[i].astype(np.float32)
            self.prev_box = box
        return kps, box


def process_sequence(seq: Sequence, extractor: PoseExtractor, half: str = "auto") -> dict:
    extractor.reset()
    kps_l, box_l, t_l, jpeg_l = [], [], [], []
    width = height = 0
    for i, frame in enumerate(iter_frames(seq.source, half)):
        height, width = frame.shape[:2]
        t0 = time.perf_counter()
        kps, box = extractor(frame)
        t_l.append((time.perf_counter() - t0) * 1000)
        kps_l.append(kps)
        box_l.append(box)
        if i % 30 == 0:  # coste de guardar la imagen comprimida (no se guarda)
            ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                jpeg_l.append(len(enc))
        del frame  # el fotograma se descarta: no se conserva ninguna imagen
    n = len(kps_l)
    if n == 0:
        raise RuntimeError(f"No se pudieron leer fotogramas de {seq.source}")
    return dict(
        kps=np.stack(kps_l).astype(np.float32), box=np.stack(box_l).astype(np.float32),
        labels=seq.frame_labels(n), fps=seq.fps, width=width, height=height,
        is_fall=seq.is_fall, group=seq.group, dataset=seq.dataset, seq_id=seq.seq_id,
        pose_ms=float(np.median(t_l[1:] if n > 1 else t_l)),
        jpeg_bytes=float(np.mean(jpeg_l)) if jpeg_l else 0.0,
    )


def skeleton_path(seq_id: str) -> Path:
    ds, rest = seq_id.split("/", 1)
    return C.SKELETON_DIR / ds / (rest.replace("/", "__") + ".npz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["urfd", "le2i"], choices=["urfd", "le2i"])
    ap.add_argument("--model", default=C.POSE_MODEL)
    ap.add_argument("--device", default=None, help="'0' para la primera GPU, 'cpu' para CPU")
    ap.add_argument("--half", default="auto", choices=["auto", "left", "right", "none"],
                    help="mitad a usar si el vídeo muestra dos imágenes lado a lado")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    extractor = PoseExtractor(args.model, args.device)
    for ds in args.dataset:
        seqs = index_dataset(ds)
        print(f"[{ds}] {len(seqs)} secuencias")
        for k, seq in enumerate(seqs, 1):
            out = skeleton_path(seq.seq_id)
            if out.exists() and not args.overwrite:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                d = process_sequence(seq, extractor, args.half)
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR en {seq.seq_id}: {e}")
                continue
            np.savez_compressed(out, **d)
            det = (d["kps"][:, :, 2].max(1) > 0).mean()
            print(f"  [{k}/{len(seqs)}] {seq.seq_id}: {len(d['kps'])} frames, "
                  f"persona detectada en {100 * det:.0f}%, pose {d['pose_ms']:.1f} ms/frame")


if __name__ == "__main__":
    main()
