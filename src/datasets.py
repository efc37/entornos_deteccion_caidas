"""Indexado de los datasets URFD y Le2i y generación de etiquetas por frame.

Ninguno de los datos originales se incluye en el repositorio (licencias). Este
módulo solo localiza los ficheros descargados por el usuario en ``data/raw`` y
construye, para cada secuencia, una etiqueta por frame:

    0 = normal / actividad cotidiana
    1 = transición de caída (el cuerpo está cayendo)
    2 = en el suelo tras la caída

Uso:  python -m src.datasets --check
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".mpg"}
IMG_EXT = {".png", ".jpg", ".jpeg"}


@dataclass
class Sequence:
    seq_id: str          # identificador único, p. ej. "urfd/fall-01"
    dataset: str         # "urfd" | "le2i"
    group: str           # agrupación para validación cruzada (secuencia o escena)
    source: Path         # vídeo o carpeta de imágenes
    fps: float
    is_fall: bool
    fall_span: tuple | None = None   # (inicio, fin) en frames 0-indexados (Le2i)
    urfd_labels: dict | None = None  # frame(0-idx) -> etiqueta original URFD

    def frame_labels(self, n_frames: int) -> np.ndarray:
        """Etiqueta por frame (0/1/2) para una secuencia de ``n_frames``."""
        y = np.zeros(n_frames, dtype=np.int8)
        if not self.is_fall:
            return y
        if self.dataset == "le2i" and self.fall_span is not None:
            s, e = self.fall_span
            s, e = max(0, s), min(n_frames - 1, e)
            y[s:e + 1] = C.LBL_FALLING
            y[e + 1:] = C.LBL_LYING_AFTER
            return y
        if self.dataset == "urfd" and self.urfd_labels:
            frames = np.array(sorted(self.urfd_labels))
            vals = np.array([self.urfd_labels[f] for f in frames])
            # cada frame toma la etiqueta del frame etiquetado más cercano
            idx = np.searchsorted(frames, np.arange(n_frames))
            idx = np.clip(idx, 0, len(frames) - 1)
            prev = np.clip(idx - 1, 0, len(frames) - 1)
            use_prev = np.abs(frames[prev] - np.arange(n_frames)) < np.abs(frames[idx] - np.arange(n_frames))
            raw = np.where(use_prev, vals[prev], vals[idx])
            # URFD: -1 no tumbado, 0 cayendo, 1 tumbado en el suelo
            y[:] = C.LBL_NORMAL
            y[raw == 0] = C.LBL_FALLING
            falling = np.where(raw == 0)[0]
            if len(falling):
                after = (raw == 1) & (np.arange(n_frames) > falling[0])
                y[after] = C.LBL_LYING_AFTER
            else:  # sin frames de transición: se marca el primer tramo tumbado
                lying = np.where(raw == 1)[0]
                if len(lying):
                    s = lying[0]
                    y[max(0, s - int(0.8 * self.fps)):s] = C.LBL_FALLING
                    y[s:] = np.where(raw[s:] == 1, C.LBL_LYING_AFTER, C.LBL_NORMAL)
            return y
        return y


# ---------------------------------------------------------------------------
# URFD
# ---------------------------------------------------------------------------
def _read_urfd_csv(path: Path) -> dict[str, dict[int, int]]:
    """Lee urfall-cam0-{falls,adls}.csv -> {secuencia: {frame0: etiqueta}}."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, header=None)
    if not re.match(r"^(fall|adl)-\d+", str(df.iloc[0, 0])):   # fichero con cabecera
        df = pd.read_csv(path)
    df = df.iloc[:, :3]
    df.columns = ["seq", "frame", "label"]
    out: dict[str, dict[int, int]] = {}
    for seq, g in df.groupby("seq"):
        out[str(seq)] = {int(f) - 1: int(l) for f, l in zip(g["frame"], g["label"])}
    return out


def _find_urfd_source(seq: str, root: Path) -> Path | None:
    for cand in [root / f"{seq}-cam0-rgb", root / seq / f"{seq}-cam0-rgb", root / seq]:
        if cand.is_dir():
            imgs = [p for p in cand.rglob("*") if p.suffix.lower() in IMG_EXT]
            if imgs:
                return sorted(imgs)[0].parent
    for cand in root.rglob(f"{seq}-cam0*"):
        if cand.suffix.lower() in VIDEO_EXT:
            return cand
    return None


def index_urfd(root: Path = C.URFD_RAW) -> list[Sequence]:
    labels = _read_urfd_csv(root / "urfall-cam0-falls.csv")
    seqs = []
    names = [f"fall-{i:02d}" for i in range(1, C.URFD_N_FALLS + 1)] + \
            [f"adl-{i:02d}" for i in range(1, C.URFD_N_ADLS + 1)]
    for name in names:
        src = _find_urfd_source(name, root)
        if src is None:
            continue
        is_fall = name.startswith("fall")
        seqs.append(Sequence(
            seq_id=f"urfd/{name}", dataset="urfd",
            # URFD no publica el sujeto de cada secuencia: se agrupa por secuencia
            group=name, source=src, fps=C.URFD_FPS, is_fall=is_fall,
            urfd_labels=labels.get(name) if is_fall else None,
        ))
    return seqs


# ---------------------------------------------------------------------------
# Le2i
# ---------------------------------------------------------------------------
def _parse_le2i_annotation(path: Path) -> tuple[int, int] | None:
    """Las dos primeras líneas son el frame de inicio y fin de la caída (1-indexado).
    "0 0" (o ausencia de caída) indica una secuencia sin caída."""
    try:
        lines = [l.strip() for l in path.read_text(errors="ignore").splitlines() if l.strip()]
    except OSError:
        return None
    if len(lines) < 2 or "," in lines[0] or "," in lines[1]:
        return None
    try:
        s, e = int(float(lines[0])), int(float(lines[1]))
    except ValueError:
        return None
    return s, e


def _le2i_scene(video: Path, root: Path) -> str:
    """Escena = primera carpeta ascendente que no se llame 'Videos'."""
    for parent in video.parents:
        if parent == root:
            break
        if parent.name.lower() not in {"videos", "video"}:
            return re.sub(r"\s+", "_", parent.name)
    return "unknown"


def index_le2i(root: Path = C.LE2I_RAW) -> list[Sequence]:
    seqs = []
    if not root.exists():
        return seqs
    for video in sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT):
        ann = None
        base = video.parent.parent if video.parent.name.lower() in {"videos", "video"} else video.parent
        for d in [base] + [p for p in base.iterdir() if p.is_dir() and p.name.lower().startswith("annotation")]:
            cand = d / f"{video.stem}.txt"
            if cand.exists():
                ann = cand
                break
        if ann is None:
            continue   # sin anotación no puede evaluarse (p. ej. algunas escenas Office/Lecture)
        span = _parse_le2i_annotation(ann)
        if span is None:
            continue
        s, e = span
        is_fall = s > 0 and e > s
        scene = _le2i_scene(video, root)
        seqs.append(Sequence(
            seq_id=f"le2i/{scene}/{re.sub(r'[^A-Za-z0-9_-]+', '_', video.stem).strip('_')}",
            dataset="le2i", group=scene, source=video, fps=C.LE2I_FPS, is_fall=is_fall,
            fall_span=(s - 1, e - 1) if is_fall else None,
        ))
    # elimina duplicados (algunos espejos de Kaggle repiten carpetas)
    uniq = {}
    for s in seqs:
        uniq.setdefault(s.seq_id, s)
    return list(uniq.values())


def index_dataset(name: str) -> list[Sequence]:
    return {"urfd": index_urfd, "le2i": index_le2i}[name]()


def summary(seqs: list[Sequence]) -> str:
    if not seqs:
        return "  (no se ha encontrado ninguna secuencia)"
    n_fall = sum(s.is_fall for s in seqs)
    groups = sorted({s.group for s in seqs})
    g = ", ".join(groups) if len(groups) <= 12 else f"{len(groups)} grupos"
    return f"  {len(seqs)} secuencias ({n_fall} con caída, {len(seqs) - n_fall} ADL) · grupos: {g}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Comprueba los datos descargados")
    ap.add_argument("--check", action="store_true")
    ap.parse_args()
    for name in ["urfd", "le2i"]:
        seqs = index_dataset(name)
        print(f"[{name}] {C.RAW_DIR / name}")
        print(summary(seqs))
        if name == "urfd" and seqs and not (C.URFD_RAW / "urfall-cam0-falls.csv").exists():
            print("  AVISO: falta urfall-cam0-falls.csv (etiquetas por frame de las caídas)")
