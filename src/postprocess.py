"""Lógica de decisión temporal compartida por la evaluación y el servicio.

Niveles de decisión (ablación de falsas alarmas):
  raw      alarma en cuanto una ventana supera el umbral de probabilidad
  persist  la probabilidad debe superar el umbral en PERSIST_K ventanas seguidas
  confirm  persist + comprobación de inactividad: durante INACTIVITY_S tras el
           candidato la persona debe seguir "en el suelo" (tronco horizontal o
           silueta más ancha que alta). Si deja de verse, se confirma por
           seguridad (posible caída fuera del campo de visión).
"""
from __future__ import annotations

import warnings

import numpy as np

from . import config as C

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

LEVELS = ["raw", "persist", "confirm"]


def torso_angle(xy: np.ndarray) -> np.ndarray:
    """Ángulo del tronco respecto a la vertical por frame (grados). xy (T,17,2)."""
    sh = np.nanmean(xy[:, [C.L_SH, C.R_SH]], axis=1)
    hp = np.nanmean(xy[:, [C.L_HIP, C.R_HIP]], axis=1)
    v = sh - hp
    return np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1] + 1e-9))


def aspect_ratio(xy: np.ndarray) -> np.ndarray:
    w = np.nanmax(xy[..., 0], 1) - np.nanmin(xy[..., 0], 1)
    h = np.nanmax(xy[..., 1], 1) - np.nanmin(xy[..., 1], 1)
    return h / np.maximum(w, 1e-3)


def down_fraction(xy: np.ndarray) -> float | None:
    """Fracción de frames visibles con la persona 'en el suelo'. None si no se ve."""
    if len(xy) == 0:
        return None
    ang, ar = torso_angle(xy), aspect_ratio(xy)
    vis = np.isfinite(ang) | np.isfinite(ar)
    if vis.sum() == 0:
        return None
    down = (np.nan_to_num(ang, nan=0) > 50) | (np.nan_to_num(ar, nan=9) < 1.0)
    return float(down[vis].mean())


def confirm_down(xy_after: np.ndarray, fps: float = C.TARGET_FPS) -> bool:
    """Confirma la caída si la persona permanece en el suelo tras el candidato."""
    if len(xy_after) < 0.5 * fps:   # fin de la grabación: no hay evidencia en contra
        return True
    f = down_fraction(xy_after)
    return True if f is None else f >= C.DOWN_FRAC


def simulate_alarms(xy: np.ndarray, t_end: np.ndarray, prob: np.ndarray, missing: np.ndarray,
                    level: str, thr: float = C.PROB_THRESHOLD, fps: float = C.TARGET_FPS) -> list[int]:
    """Simula el servicio sobre una secuencia completa. Devuelve los frames de alarma."""
    alarms: list[int] = []
    run, last = 0, -10 ** 9
    k = 1 if level == "raw" else C.PERSIST_K
    refr = int(C.REFRACTORY_S * fps)
    n_inact = int(C.INACTIVITY_S * fps)
    for t, p, miss in zip(t_end, prob, missing):
        above = (p >= thr) and (miss <= C.MAX_MISSING_FRAC)
        run = run + 1 if above else 0
        if run >= k and t - last > refr:
            if level == "confirm":
                if not confirm_down(xy[t + 1: t + 1 + n_inact], fps):
                    continue
                t_alarm = int(min(t + n_inact, len(xy) - 1))
            else:
                t_alarm = int(t)
            alarms.append(t_alarm)
            last, run = t_alarm, 0
    return alarms
