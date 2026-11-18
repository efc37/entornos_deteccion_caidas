"""Limpieza, remuestreo temporal, ventaneo y normalización de esqueletos.

Pasos (para cada secuencia):
 1. Puntos con confianza < KP_CONF_MIN -> no visibles (NaN).
 2. Interpolación lineal de huecos cortos (<= MAX_GAP_FRAMES) por articulación.
 3. Remuestreo a TARGET_FPS (25 fps) para que una ventana dure lo mismo en
    URFD (30 fps) y Le2i (25 fps): imprescindible para la evaluación cruzada.
 4. Ventanas deslizantes de WINDOW frames con etiqueta por ventana:
      positiva  si contiene >= MIN_FALL_FRAMES frames de transición de caída,
      ignorada  si solapa parcialmente la caída o el tramo en el suelo posterior,
      negativa  en otro caso.
 5. Normalización por ventana (invarianza a posición y escala): origen en el
    centro de la cadera del primer frame y escala = longitud mediana del torso.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from pathlib import Path

import numpy as np

from . import config as C

# nanmean/nanmedian sobre tramos sin persona visible generan avisos inofensivos
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load_records(dataset: str, skeleton_dir: Path = C.SKELETON_DIR) -> list[dict]:
    files = sorted((skeleton_dir / dataset).glob("*.npz"))
    recs = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            recs.append({k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files})
    return recs


# ---------------------------------------------------------------------------
# Limpieza y remuestreo
# ---------------------------------------------------------------------------
def _interp_gaps(a: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpola linealmente huecos NaN de longitud <= max_gap en un vector 1D."""
    a = a.copy()
    nan = np.isnan(a)
    if nan.all() or not nan.any():
        return a
    idx = np.arange(len(a))
    valid = ~nan
    filled = np.interp(idx, idx[valid], a[valid])
    # longitud de cada tramo NaN
    run_start = np.where(nan & ~np.r_[False, nan[:-1]])[0]
    run_end = np.where(nan & ~np.r_[nan[1:], False])[0]
    for s, e in zip(run_start, run_end):
        inner = s > 0 and e < len(a) - 1
        if inner and (e - s + 1) <= max_gap:
            a[s:e + 1] = filled[s:e + 1]
    return a


def clean(kps: np.ndarray, conf_min: float = C.KP_CONF_MIN, max_gap: int = C.MAX_GAP_FRAMES):
    """kps (T,17,3) -> xy (T,17,2) con NaN en puntos no visibles, conf (T,17)."""
    xy = kps[..., :2].astype(np.float32).copy()
    conf = kps[..., 2].astype(np.float32).copy()
    xy[conf < conf_min] = np.nan
    T, V, _ = xy.shape
    for v in range(V):
        for d in range(2):
            xy[:, v, d] = _interp_gaps(xy[:, v, d], max_gap)
    conf = np.where(np.isnan(xy[..., 0]), 0.0, np.maximum(conf, conf_min)).astype(np.float32)
    return xy, conf


def resample(xy: np.ndarray, conf: np.ndarray, labels: np.ndarray, fps: float,
             target: float = C.TARGET_FPS):
    """Remuestrea linealmente a ``target`` fps (las etiquetas por vecino más cercano)."""
    if abs(fps - target) < 1e-3:
        return xy, conf, labels
    T = len(xy)
    t_new = np.arange(0, (T - 1) / fps + 1e-9, 1.0 / target)
    pos = t_new * fps
    i0 = np.clip(np.floor(pos).astype(int), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    w = (pos - i0)[:, None, None]
    xy_n = (1 - w) * xy[i0] + w * xy[i1]
    conf_n = np.minimum(conf[i0], conf[i1])
    lab_n = labels[np.clip(np.rint(pos).astype(int), 0, T - 1)]
    return xy_n.astype(np.float32), conf_n.astype(np.float32), lab_n.astype(np.int8)


@dataclass
class PreparedSeq:
    seq_id: str
    dataset: str
    group: str
    is_fall: bool
    xy: np.ndarray        # (T,17,2) a TARGET_FPS, NaN si no visible
    conf: np.ndarray      # (T,17)
    labels: np.ndarray    # (T,)
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.xy) / C.TARGET_FPS

    def fall_span(self) -> tuple[int, int] | None:
        idx = np.where(self.labels == C.LBL_FALLING)[0]
        return (int(idx[0]), int(idx[-1])) if len(idx) else None


def prepare(records: list[dict]) -> list[PreparedSeq]:
    out = []
    for r in records:
        xy, conf = clean(r["kps"])
        xy, conf, lab = resample(xy, conf, r["labels"].astype(np.int8), float(r["fps"]))
        out.append(PreparedSeq(
            seq_id=str(r["seq_id"]), dataset=str(r["dataset"]), group=str(r["group"]),
            is_fall=bool(r["is_fall"]), xy=xy, conf=conf, labels=lab,
            meta=dict(pose_ms=float(r.get("pose_ms", 0.0)), jpeg_bytes=float(r.get("jpeg_bytes", 0.0)),
                      width=int(r.get("width", 0)), height=int(r.get("height", 0)),
                      fps=float(r["fps"])),
        ))
    return out


def load_dataset(dataset: str, skeleton_dir: Path = C.SKELETON_DIR) -> list[PreparedSeq]:
    return prepare(load_records(dataset, skeleton_dir))


# ---------------------------------------------------------------------------
# Ventanas
# ---------------------------------------------------------------------------
def window_label(lab: np.ndarray, total_falling: int) -> int:
    n_fall = int((lab == C.LBL_FALLING).sum())
    if n_fall >= min(C.MIN_FALL_FRAMES, max(1, total_falling)):
        return C.WIN_POS
    if (lab != C.LBL_NORMAL).any():
        return C.WIN_IGNORE
    return C.WIN_NEG


@dataclass
class Windows:
    xy: np.ndarray        # (N, W, 17, 2)
    conf: np.ndarray      # (N, W, 17)
    y: np.ndarray         # (N,)  1 caída, 0 no caída, -1 ignorada
    seq: np.ndarray       # (N,)  índice de la secuencia en la lista
    group: np.ndarray     # (N,)  grupo para validación cruzada
    t_end: np.ndarray     # (N,)  frame (a TARGET_FPS) en que termina la ventana
    missing: np.ndarray   # (N,)  fracción de frames sin persona visible
    cache: dict = field(default_factory=dict)   # features / entradas ya calculadas

    def subset(self, m: np.ndarray) -> "Windows":
        w = Windows(*(getattr(self, f)[m] for f in
                      ["xy", "conf", "y", "seq", "group", "t_end", "missing"]))
        w.cache = {k: v[m] for k, v in self.cache.items()}
        return w

    def __len__(self):
        return len(self.y)


def make_windows(seqs: list[PreparedSeq], stride: int = C.STRIDE_TRAIN,
                 window: int = C.WINDOW, seq_offset: int = 0) -> Windows:
    xs, cs, ys, ss, gs, ts, ms = [], [], [], [], [], [], []
    for si, s in enumerate(seqs):
        T = len(s.xy)
        if T < window:
            continue
        total_falling = int((s.labels == C.LBL_FALLING).sum())
        visible = ~np.isnan(s.xy[..., 0]).all(axis=1)
        for end in range(window, T + 1, stride):
            sl = slice(end - window, end)
            xs.append(s.xy[sl])
            cs.append(s.conf[sl])
            ys.append(window_label(s.labels[sl], total_falling))
            ss.append(si + seq_offset)
            gs.append(s.group)
            ts.append(end - 1)
            ms.append(1.0 - visible[sl].mean())
    if not xs:
        z = np.zeros((0, window, C.N_KP, 2), np.float32)
        return Windows(z, z[..., 0], *(np.zeros(0) for _ in range(5)))
    return Windows(np.stack(xs).astype(np.float32), np.stack(cs).astype(np.float32),
                   np.array(ys, np.int8), np.array(ss), np.array(gs), np.array(ts),
                   np.array(ms, np.float32))


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------
def _mid(xy: np.ndarray, a: int, b: int) -> np.ndarray:
    return np.nanmean(np.stack([xy[..., a, :], xy[..., b, :]]), axis=0)


def window_scale(xy: np.ndarray) -> float:
    """Escala de una ventana (T,17,2): mediana de la longitud del torso."""
    sh = _mid(xy, C.L_SH, C.R_SH)
    hp = _mid(xy, C.L_HIP, C.R_HIP)
    torso = np.linalg.norm(sh - hp, axis=-1)
    s = np.nanmedian(torso) if np.isfinite(torso).any() else np.nan
    if not np.isfinite(s) or s < 1e-3:
        h = np.nanmax(xy[..., 1]) - np.nanmin(xy[..., 1]) if np.isfinite(xy).any() else np.nan
        s = h / 3.0 if np.isfinite(h) and h > 1e-3 else 1.0
    return float(s)


def window_origin(xy: np.ndarray) -> np.ndarray:
    hp = _mid(xy, C.L_HIP, C.R_HIP)
    valid = np.isfinite(hp).all(axis=1)
    if valid.any():
        return hp[np.argmax(valid)]
    allv = xy.reshape(-1, 2)
    allv = allv[np.isfinite(allv).all(axis=1)]
    return allv.mean(0) if len(allv) else np.zeros(2, np.float32)


def normalize_window(xy: np.ndarray) -> np.ndarray:
    """(T,17,2) -> coordenadas relativas a la cadera inicial y divididas por el torso."""
    return ((xy - window_origin(xy)) / window_scale(xy)).astype(np.float32)


def deep_input(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Tensor de entrada a las redes: (T, 17, 5) = x, y, conf, dx, dy (NaN -> 0)."""
    n = normalize_window(xy)
    d = np.zeros_like(n)
    d[1:] = n[1:] - n[:-1]
    n = np.nan_to_num(n, nan=0.0)
    d = np.nan_to_num(d, nan=0.0)
    c = np.where(np.isnan(xy[..., 0]), 0.0, conf)[..., None]
    out = np.concatenate([n, c, d], axis=-1)
    return np.clip(out, -10, 10).astype(np.float32)


def deep_inputs(w: Windows) -> np.ndarray:
    if "deep" not in w.cache:
        w.cache["deep"] = (np.stack([deep_input(x, c) for x, c in zip(w.xy, w.conf)]) if len(w)
                           else np.zeros((0, C.WINDOW, C.N_KP, 5), np.float32))
    return w.cache["deep"]
