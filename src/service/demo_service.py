"""Demostración del servicio en la casa inteligente.

Modos:
  * Reproducir una secuencia de esqueleto ya extraída (sin imágenes):
        python -m src.service.demo_service --model results/models/gru_urfd.pt \
               --seq "data/skeletons/urfd/fall-01.npz" --respuesta ninguna
  * Cámara en directo (YOLOv8-Pose en local; los fotogramas se descartan):
        python -m src.service.demo_service --model results/models/gru_urfd.pt --webcam 0
  * Sin datos reales (modelo y secuencias sintéticas, solo para probar):
        python -m src.service.demo_service --synthetic

--respuesta simula lo que contesta la persona a "¿Se encuentra bien?":
  ok (contesta 'estoy bien' a los 5 s) · ayuda (pide ayuda a los 3 s) · ninguna.
Genera results/demo/<secuencia>_eventos.json y una figura con la línea temporal.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .. import config as C
from ..preprocessing import PreparedSeq, load_dataset, prepare
from .fall_service import FallService, SmartHome

RESP = {"ok": ("ok", 5.0), "ayuda": ("help", 3.0), "ninguna": None}


def load_model(path: Path):
    if str(path).endswith(".pt"):
        from ..train_deep_model import DeepModel
        return DeepModel.load(path)
    from ..train_baseline import FeatureModel
    return FeatureModel.load(path)


def replay(service: FallService, seq: PreparedSeq, tail_s: float = C.CONFIRM_TIMEOUT_S + 2):
    for xy, cf in zip(seq.xy, seq.conf):
        service.push(xy, cf)
    # tras el final de la grabación la persona sigue en la misma postura
    for _ in range(int(tail_s * C.TARGET_FPS)):
        if service.state.value == "MONITORIZANDO" and service.frame > len(seq.xy) + C.INACTIVITY_S * C.TARGET_FPS:
            break
        service.push(seq.xy[-1], seq.conf[-1])


def plot_timeline(service: FallService, seq: PreparedSeq, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([x[0] for x in service.trace])
    p = np.array([x[1] for x in service.trace], float)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    span = seq.fall_span()
    if span:
        ax.axvspan(span[0] / C.TARGET_FPS, span[1] / C.TARGET_FPS, color="#e4e3df", label="Caída real (etiqueta)")
    m = np.isfinite(p)
    ax.plot(t[m], p[m], color="#2a78d6", lw=2, label="Probabilidad de caída")
    ax.axhline(service.thr, color="#52514e", lw=1, ls="--", label="Umbral")
    colors = {"POSIBLE_CAIDA": "#eda100", "PREGUNTA": "#6250d6", "ALERTA": "#d03b3b",
              "CANCELADA": "#1baf7a", "DESCARTADA": "#1baf7a", "FUERA_DE_VISTA": "#52514e"}
    ev = service.events
    for a, b in zip(ev, ev[1:] + [None]):
        if a.kind == "PREGUNTA":
            end = b.t if b is not None else t[-1]
            ax.axvspan(a.t, end, color="#6250d6", alpha=0.08, label="Esperando respuesta")
    for e in service.events:
        ax.axvline(e.t, color=colors.get(e.kind, "#52514e"), lw=1.5)
        ax.text(e.t, 1.04, e.kind.replace("_", " ").lower(), rotation=30, fontsize=7, color="#0b0b0b",
                ha="left", va="bottom")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("tiempo (s)", color="#52514e", fontsize=9)
    ax.set_ylabel("p(caída)", color="#52514e", fontsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(f"Servicio en funcionamiento · {seq.seq_id}", loc="left", fontsize=10, pad=34)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def run_webcam(service: FallService, cam: int, model_name: str):
    """Bucle en directo: cámara -> pose -> servicio. No se guarda ninguna imagen."""
    import cv2

    from ..extract_skeletons import PoseExtractor
    from ..preprocessing import clean
    ext = PoseExtractor(model_name)
    cap = cv2.VideoCapture(cam)
    print("Ctrl+C para terminar")
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            kps, _ = ext(frame)
            del frame                                   # privacidad: la imagen se descarta
            xy, cf = clean(kps[None], max_gap=0)
            service.push(xy[0], cf[0])
            time.sleep(max(0.0, 1.0 / C.TARGET_FPS - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--seq", type=Path, nargs="*", help="ficheros .npz de esqueleto")
    ap.add_argument("--respuesta", choices=list(RESP), default="ninguna")
    ap.add_argument("--webcam", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR / "demo")
    args = ap.parse_args()

    if args.synthetic:
        from ..synthetic import generate
        from ..preprocessing import make_windows
        from ..train_baseline import FeatureModel
        d = generate(C.DATA_DIR / "synthetic_skeletons", n_per_class=8)
        seqs = load_dataset("urfd", d)
        test = [s for s in seqs if "fall" in s.seq_id][:1] + [s for s in seqs if "lie_down" in s.seq_id][:1]
        model = FeatureModel("rf").fit(make_windows([s for s in seqs if s not in test]))
    else:
        if args.model is None:
            raise SystemExit("Indique --model (p. ej. results/models/gru_urfd.pt)")
        model = load_model(args.model)
        test = []
        for f in args.seq or []:
            with np.load(f) as z:
                rec = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
            test += prepare([rec])

    if args.webcam is not None:
        service = FallService(model, SmartHome(), responder=lambda t: RESP[args.respuesta])
        run_webcam(service, args.webcam, C.POSE_MODEL)
        service.save_log(args.out / "webcam_eventos.json")
        return

    for seq in test:
        print(f"\n=== {seq.seq_id}  (caída real: {'sí' if seq.is_fall else 'no'}) ===")
        service = FallService(model, SmartHome(), responder=lambda t: RESP[args.respuesta])
        replay(service, seq)
        name = seq.seq_id.replace("/", "__")
        service.save_log(args.out / f"{name}_eventos.json")
        plot_timeline(service, seq, args.out / f"{name}_linea_temporal.png")
    print(f"\nRegistros y figuras en {args.out}")


if __name__ == "__main__":
    main()
