"""Features geométricas y cinemáticas diseñadas a mano (método base).

Se calculan por ventana a partir del esqueleto normalizado (origen en la cadera
inicial y escala = longitud del torso), de modo que son invariantes a la posición
y al tamaño de la persona en la imagen. Recogen las señales clásicas de caída
(guía DTIC, T17): velocidad vertical, relación de aspecto y caída del centroide,
más la inclinación del tronco y la altura de la cabeza.
"""
from __future__ import annotations

import warnings

import numpy as np

from . import config as C
from .preprocessing import Windows, normalize_window

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

FEATURE_NAMES = [
    "torso_angle_first", "torso_angle_last", "torso_angle_max", "torso_angle_delta",
    "torso_angvel_max",
    "hip_dy", "hip_vy_max", "hip_vy_mean", "hip_ay_max",
    "centroid_dy", "centroid_vy_max",
    "head_dy", "head_vy_max", "head_minus_hip_last",
    "aspect_first", "aspect_last", "aspect_min", "aspect_delta",
    "height_ratio_last", "height_ratio_min",
    "joint_speed_mean", "joint_speed_max", "still_last",
    "visible_frac", "lower_visible_frac",
]


def _mid(n, a, b):
    return np.nanmean(np.stack([n[:, a], n[:, b]]), axis=0)


def _ffill(a: np.ndarray) -> np.ndarray:
    """Rellena NaN hacia delante y hacia atrás (vector 1D)."""
    a = a.copy()
    ok = np.isfinite(a)
    if not ok.any():
        return np.zeros_like(a)
    idx = np.where(ok, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    a = a[idx]
    first = np.argmax(ok)
    a[:first] = a[first]
    return a


def window_features(xy: np.ndarray, conf: np.ndarray, fps: float = C.TARGET_FPS) -> np.ndarray:
    n = normalize_window(xy)                      # (T,17,2) en torsos, y hacia abajo
    sh, hp = _mid(n, C.L_SH, C.R_SH), _mid(n, C.L_HIP, C.R_HIP)
    head = np.nanmean(n[:, C.HEAD], axis=1)
    # inclinación del tronco respecto a la vertical (0 = erguido, 90 = horizontal)
    v = sh - hp
    ang = _ffill(np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1] + 1e-9)))
    ang = np.clip(ang, 0, 180)
    hip_y, head_y = _ffill(hp[:, 1]), _ffill(head[:, 1])
    cen_y = _ffill(np.nanmean(n[:, :, 1], axis=1))
    vy_hip = np.gradient(hip_y) * fps             # torsos/s (positivo = hacia abajo)
    vy_cen = np.gradient(cen_y) * fps
    vy_head = np.gradient(head_y) * fps
    # bbox del esqueleto
    w = _ffill(np.nanmax(n[:, :, 0], 1) - np.nanmin(n[:, :, 0], 1))
    h = _ffill(np.nanmax(n[:, :, 1], 1) - np.nanmin(n[:, :, 1], 1))
    aspect = h / np.maximum(w, 1e-3)
    h0 = np.nanmax(h[: max(3, len(h) // 4)]) if np.isfinite(h).any() else 1.0
    hr = h / max(h0, 1e-3)
    # velocidad media de las articulaciones
    d = np.linalg.norm(np.diff(n, axis=0), axis=-1) * fps          # (T-1,17)
    spd = np.nan_to_num(np.nanmean(d, axis=1))
    vis = np.isfinite(n[..., 0])
    T = len(n)
    k = max(3, T // 4)
    f = [
        ang[0], ang[-1], ang.max(), ang[-1] - ang[0], np.abs(np.gradient(ang) * fps).max(),
        hip_y[-1] - hip_y[0], vy_hip.max(), vy_hip.mean(), np.abs(np.gradient(vy_hip) * fps).max(),
        cen_y[-1] - cen_y[0], vy_cen.max(),
        head_y[-1] - head_y[0], vy_head.max(), head_y[-1] - hip_y[-1],
        aspect[0], aspect[-1], aspect.min(), aspect[-1] - aspect[0],
        hr[-1], hr.min(),
        spd.mean(), spd.max(), spd[-k:].mean(),
        vis.mean(), vis[:, C.LOWER_BODY].mean(),
    ]
    return np.nan_to_num(np.array(f, np.float32), nan=0.0, posinf=50.0, neginf=-50.0)


def features(w: Windows) -> np.ndarray:
    if "feat" not in w.cache:
        w.cache["feat"] = (np.stack([window_features(x, c) for x, c in zip(w.xy, w.conf)]) if len(w)
                           else np.zeros((0, len(FEATURE_NAMES)), np.float32))
    return w.cache["feat"]


def rule_score(F: np.ndarray) -> np.ndarray:
    """Baseline sin entrenamiento: puntuación en [0,1] = fracción de reglas cumplidas
    (tronco inclinado al final, descenso rápido de la cadera y bbox más ancho que alto)."""
    i = {n: k for k, n in enumerate(FEATURE_NAMES)}
    r1 = F[:, i["torso_angle_last"]] > C.RULE_TORSO_ANGLE
    r2 = F[:, i["hip_vy_max"]] > C.RULE_DESCENT_SPEED
    r3 = F[:, i["aspect_last"]] < C.RULE_ASPECT
    return (r1.astype(float) + r2 + r3) / 3.0
