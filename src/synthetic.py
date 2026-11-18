"""Generador de secuencias de esqueleto sintéticas (SOLO para pruebas).

Permite ejecutar y verificar todo el pipeline (preprocesado, modelos, métricas,
servicio) sin descargar los datasets. Los resultados con datos sintéticos NO
tienen valor científico: no deben aparecer en el póster.

Genera actividades con la misma estructura de ficheros que extract_skeletons:
caídas rápidas y ADL difíciles (tumbarse despacio, sentarse, agacharse, andar).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as C

# Cuerpo erguido en coordenadas locales (unidades = altura de la persona), y hacia abajo
_UPPER = {0: (0.0, -0.62), 1: (0.03, -0.64), 2: (-0.03, -0.64), 3: (0.06, -0.62), 4: (-0.06, -0.62),
          5: (0.12, -0.45), 6: (-0.12, -0.45), 7: (0.15, -0.25), 8: (-0.15, -0.25),
          9: (0.15, -0.05), 10: (-0.15, -0.05), 11: (0.08, 0.0), 12: (-0.08, 0.0)}


def _pose(hip, theta, thigh, shank, H, facing=1.0):
    """Esqueleto (17,2) a partir de la cadera, inclinación del tronco y piernas (rad)."""
    pts = np.zeros((17, 2))
    ct, st = np.cos(theta), np.sin(theta)
    for k, (x, y) in _UPPER.items():
        x *= facing
        pts[k] = hip + H * np.array([ct * x - st * y * facing, st * x * facing + ct * y])
    for side, (h, kn, an) in enumerate([(11, 13, 15), (12, 14, 16)]):
        knee = pts[h] + 0.25 * H * np.array([np.sin(thigh) * facing, np.cos(thigh)])
        pts[kn] = knee
        pts[an] = knee + 0.25 * H * np.array([np.sin(shank) * facing, np.cos(shank)])
    return pts


def _activity(kind: str, fps: float, rng: np.random.Generator, H: float, floor: float, x0: float):
    """Devuelve (lista de poses, etiquetas por frame)."""
    f = lambda s: max(2, int(round(s * fps)))  # noqa: E731
    facing = rng.choice([-1.0, 1.0])
    stand = dict(hip_y=floor - 0.5 * H, theta=0.0, thigh=0.0, shank=0.0)
    poses, labels = [], []

    def seg(n, start, end, x_start, x_end, lab):
        for i, t in enumerate(np.linspace(0, 1, n)):
            g = {k: start[k] + (end[k] - start[k]) * t for k in start}
            x = x_start + (x_end - x_start) * t
            poses.append(_pose(np.array([x, g["hip_y"]]), g["theta"], g["thigh"], g["shank"], H, facing))
            labels.append(lab)

    walk = rng.uniform(-0.6, 0.6) * H
    pre = f(rng.uniform(1.5, 3.0))
    seg(pre, stand, stand, x0, x0 + walk, C.LBL_NORMAL)
    x = x0 + walk
    if kind == "fall":
        lying = dict(hip_y=floor - 0.08 * H, theta=np.pi / 2 * rng.uniform(0.85, 1.0),
                     thigh=np.pi / 2 * 0.9, shank=np.pi / 2 * 0.9)
        seg(f(rng.uniform(0.5, 0.9)), stand, lying, x, x + 0.4 * H * facing, C.LBL_FALLING)
        seg(f(rng.uniform(2.5, 4.0)), lying, lying, x + 0.4 * H * facing, x + 0.4 * H * facing, C.LBL_LYING_AFTER)
    elif kind == "lie_down":   # ADL difícil: tumbarse despacio pasando por sentado
        sit = dict(hip_y=floor - 0.25 * H, theta=0.15, thigh=np.pi / 2, shank=0.0)
        lying = dict(hip_y=floor - 0.25 * H, theta=np.pi / 2, thigh=np.pi / 2, shank=0.2)
        seg(f(rng.uniform(1.2, 2.0)), stand, sit, x, x, C.LBL_NORMAL)
        seg(f(rng.uniform(1.5, 2.5)), sit, lying, x, x, C.LBL_NORMAL)
        seg(f(2.5), lying, lying, x, x, C.LBL_NORMAL)
    elif kind == "sit":
        sit = dict(hip_y=floor - 0.25 * H, theta=0.1, thigh=np.pi / 2, shank=0.0)
        seg(f(rng.uniform(0.8, 1.5)), stand, sit, x, x, C.LBL_NORMAL)
        seg(f(2.0), sit, sit, x, x, C.LBL_NORMAL)
        seg(f(1.0), sit, stand, x, x, C.LBL_NORMAL)
    elif kind == "bend":       # recoger un objeto del suelo
        bend = dict(hip_y=floor - 0.42 * H, theta=np.radians(rng.uniform(60, 85)), thigh=0.3, shank=-0.2)
        seg(f(rng.uniform(0.8, 1.2)), stand, bend, x, x, C.LBL_NORMAL)
        seg(f(1.0), bend, bend, x, x, C.LBL_NORMAL)
        seg(f(1.0), bend, stand, x, x, C.LBL_NORMAL)
    else:                      # andar
        seg(f(3.0), stand, stand, x, x + rng.uniform(-1.0, 1.0) * H, C.LBL_NORMAL)
    return np.array(poses), np.array(labels, np.int8)


def make_sequence(kind: str, dataset: str, rng: np.random.Generator) -> dict:
    fps = C.URFD_FPS if dataset == "urfd" else C.LE2I_FPS
    W, Hh = (640, 480) if dataset == "urfd" else (320, 240)
    H = rng.uniform(0.55, 0.75) * Hh
    floor = Hh * rng.uniform(0.85, 0.95)
    poses, labels = _activity(kind, fps, rng, H, floor, rng.uniform(0.3, 0.7) * W)
    T = len(poses)
    xy = poses + rng.normal(0, 0.008 * H, poses.shape)
    conf = rng.uniform(0.6, 1.0, (T, 17))
    if dataset == "le2i" and rng.random() < 0.3:   # oclusión parcial por mobiliario
        s = rng.integers(0, T // 2)
        conf[s:s + T // 3][:, C.LOWER_BODY] = 0.1
    if rng.random() < 0.3:                          # frames sin detección
        s = rng.integers(0, T - 5)
        conf[s:s + rng.integers(2, 6)] = 0.0
    kps = np.concatenate([xy, conf[..., None]], -1).astype(np.float32)
    kps[conf < 0.05] = 0.0
    return dict(kps=kps, box=np.zeros((T, 4), np.float32), labels=labels, fps=fps, width=W, height=Hh,
                is_fall=kind == "fall", pose_ms=12.0, jpeg_bytes=40000.0 if dataset == "urfd" else 12000.0)


def generate(out_dir: Path, n_per_class: int = 10, seed: int = C.SEED) -> Path:
    """Crea out_dir/{urfd,le2i}/*.npz con secuencias sintéticas."""
    rng = np.random.default_rng(seed)
    kinds = ["fall", "fall", "lie_down", "sit", "bend", "walk"]
    for dataset in ["urfd", "le2i"]:
        d = out_dir / dataset
        d.mkdir(parents=True, exist_ok=True)
        scenes = ["Home_01", "Home_02", "Coffee_room_01", "Office"]
        k = 0
        for kind in kinds:
            for i in range(n_per_class):
                rec = make_sequence(kind, dataset, rng)
                group = f"{kind}-{i:02d}" if dataset == "urfd" else scenes[k % len(scenes)]
                sid = f"{dataset}/{kind}-{i:02d}-{k}"
                rec.update(group=group, dataset=dataset, seq_id=sid)
                np.savez_compressed(d / f"{kind}-{i:02d}-{k}.npz", **rec)
                k += 1
    return out_dir
