"""Métricas por ventana y por evento.

* Por ventana: sensibilidad, especificidad, precisión, F1, exactitud equilibrada,
  ROC-AUC y AP. Se excluyen las ventanas ignoradas y las de persona no visible
  (éstas se cuentan aparte: el servicio las trata como "fuera del campo de visión").
* Por evento (lo que vería la casa inteligente): cada secuencia se recorre en
  streaming. Una caída se detecta si hay alarma en [inicio-1 s, fin+4 s (+ espera
  de confirmación)]; cualquier otra alarma es una falsa alarma. Se informa de la
  sensibilidad por evento, falsas alarmas por hora de actividad normal y latencia.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .postprocess import LEVELS, simulate_alarms
from .preprocessing import PreparedSeq, make_windows


def window_metrics(y: np.ndarray, p: np.ndarray, thr: float = C.PROB_THRESHOLD) -> dict:
    y = y.astype(int)
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (tp and prec + sens) else 0.0
    both = 0 < y.sum() < len(y)
    return dict(
        n_pos=int(y.sum()), n_neg=int((y == 0).sum()), tp=tp, tn=tn, fp=fp, fn=fn,
        sensitivity=sens, specificity=spec, precision=prec, f1=f1,
        balanced_acc=np.nanmean([sens, spec]),
        roc_auc=roc_auc_score(y, p) if both else np.nan,
        avg_precision=average_precision_score(y, p) if both else np.nan,
    )


def stream_predictions(model, seqs: list[PreparedSeq]) -> list[dict]:
    """Probabilidad por ventana (paso STRIDE_STREAM) para cada secuencia completa."""
    w = make_windows(seqs, stride=C.STRIDE_STREAM)
    p = model.predict_proba(w)
    out = []
    for si, s in enumerate(seqs):
        m = w.seq == si
        out.append(dict(seq=s, t_end=w.t_end[m], prob=p[m], missing=w.missing[m]))
    return out


def event_metrics(streams: list[dict], level: str, thr: float = C.PROB_THRESHOLD) -> dict:
    fps = C.TARGET_FPS
    tp = fn = fp = 0
    normal_s = 0.0
    lat = []
    tol0, tol1 = (int(t * fps) for t in C.EVENT_TOLERANCE_S)
    wait = int(C.INACTIVITY_S * fps) if level == "confirm" else 0
    per_seq = []
    for st in streams:
        s: PreparedSeq = st["seq"]
        alarms = simulate_alarms(s.xy, st["t_end"], st["prob"], st["missing"], level, thr)
        span = s.fall_span() if s.is_fall else None
        if span is None:
            fp += len(alarms)
            normal_s += s.duration_s
            hit = None
        else:
            lo, hi = span[0] - tol0, span[1] + tol1 + wait
            hits = [a for a in alarms if lo <= a <= hi]
            fp += len(alarms) - len(hits)
            normal_s += max(0, span[0] - tol0) / fps
            hit = bool(hits)
            if hits:
                tp += 1
                lat.append((hits[0] - span[0]) / fps)
            else:
                fn += 1
        per_seq.append(dict(seq_id=s.seq_id, is_fall=s.is_fall, n_alarms=len(alarms), detected=hit))
    hours = normal_s / 3600.0
    return dict(
        level=level, falls=tp + fn, detected=tp,
        event_sensitivity=tp / (tp + fn) if tp + fn else np.nan,
        false_alarms=fp, normal_hours=hours,
        false_alarms_per_hour=fp / hours if hours > 0 else np.nan,
        latency_s=float(np.mean(lat)) if lat else np.nan, per_seq=per_seq,
    )


def all_event_metrics(streams: list[dict], thr: float = C.PROB_THRESHOLD) -> list[dict]:
    return [event_metrics(streams, lv, thr) for lv in LEVELS]
