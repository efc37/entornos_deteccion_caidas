"""Ejecución de todos los experimentos y generación de tablas y figuras.

Protocolos
  E1  URFD: validación cruzada agrupada por secuencia (5 particiones).
  E2  Le2i: validación "leave-one-scene-out" (una escena entera en test).
  E3  Entre datasets: entrenar en URFD -> probar en Le2i y viceversa, sin reajuste.
  E4  Por evento (streaming): falsas alarmas/hora y sensibilidad según el nivel de
      decisión (raw / persistencia / persistencia + confirmación de inactividad).
  E5  Campo de visión y oclusiones: rendimiento al ocultar articulaciones al azar
      o la parte inferior del cuerpo; rendimiento por escena (Le2i).
  E6  Privacidad y coste: bytes por segundo de cada representación y tiempos.

Modelos: reglas (sin entrenamiento), Random Forest, SVM (método base) frente a
GRU y ST-GCN sobre la secuencia de esqueleto (propuesta), con y sin aumento.

Uso:
    python -m src.experiments                    # todos los datasets disponibles
    python -m src.experiments --models rules rf gru --quick
    python -m src.experiments --synthetic        # prueba del pipeline sin datos reales
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from . import config as C
from .evaluation import all_event_metrics, stream_predictions, window_metrics
from .preprocessing import PreparedSeq, Windows, load_dataset, make_windows
from .train_baseline import FeatureModel, trainable

warnings.filterwarnings("ignore", category=UserWarning)

ALL_MODELS = ["rules", "rf", "svm", "gru", "gru_noaug", "stgcn"]
MAIN_METRICS = ["sensitivity", "specificity", "precision", "f1", "balanced_acc", "roc_auc"]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def make_model(key: str):
    if key in ("rules", "rf", "svm"):
        return FeatureModel(key)
    from .train_deep_model import DeepModel
    return {"gru": lambda: DeepModel("gru", True), "gru_noaug": lambda: DeepModel("gru", False),
            "stgcn": lambda: DeepModel("stgcn", True)}[key]()


MODEL_LABEL = {"rules": "Reglas", "rf": "Random Forest", "svm": "SVM", "gru": "GRU",
               "gru_noaug": "GRU sin aumento", "stgcn": "ST-GCN"}


# ---------------------------------------------------------------------------
# Particiones
# ---------------------------------------------------------------------------
def seq_splits(seqs: list[PreparedSeq], n_splits: int = 5):
    groups = np.array([s.group for s in seqs])
    y = np.array([s.is_fall for s in seqs], int)
    n_groups = len(set(groups))
    if n_groups <= 8:   # pocas escenas -> dejar una escena fuera cada vez
        return list(LeaveOneGroupOut().split(np.zeros(len(seqs)), y, groups)), "leave-one-group-out"
    k = min(n_splits, n_groups)
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=C.SEED)
    return list(sgk.split(np.zeros(len(seqs)), y, groups)), f"{k}-fold agrupado"


def occlude(w: Windows, kind: str, q: float, rng: np.random.Generator) -> Windows:
    """Copia de las ventanas con articulaciones ocultas (sin caché)."""
    xy, conf = w.xy.copy(), w.conf.copy()
    N, T, V, _ = xy.shape
    if kind == "random":
        mask = rng.random((N, T, V)) < q
    else:  # piernas ocultas durante la fracción final q de la ventana (p. ej. tras un sofá)
        mask = np.zeros((N, T, V), bool)
        n = int(round(q * T))
        if n:
            mask[:, T - n:, C.LOWER_BODY] = True
    xy[mask] = np.nan
    # igual que en el pipeline real: los huecos cortos interiores se interpolan
    flat = pd.DataFrame(xy.transpose(1, 0, 2, 3).reshape(T, -1))
    flat = flat.interpolate(limit=C.MAX_GAP_FRAMES, limit_area="inside")
    xy = flat.to_numpy(np.float32).reshape(T, N, V, 2).transpose(1, 0, 2, 3).copy()
    conf = np.where(np.isnan(xy[..., 0]), 0.0, np.where(mask, C.KP_CONF_MIN, conf)).astype(np.float32)
    return Windows(xy, conf, w.y, w.seq, w.group, w.t_end, w.missing)


OCC_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]


# ---------------------------------------------------------------------------
# Un protocolo = lista de (train_seqs, test_seqs)
# ---------------------------------------------------------------------------
def run_protocol(name: str, folds: list[tuple[list, list]], models: list[str], log, occlusion: bool):
    rows_w, rows_e, rows_g, rows_o, rows_t = [], [], [], [], []
    rng = np.random.default_rng(C.SEED)
    for key in models:
        t0 = time.perf_counter()
        ys, ps, streams, fold_m = [], [], [], []
        occ = {(k, q): ([], []) for k in ("random", "lower") for q in OCC_LEVELS}
        infer_ms = []
        for fi, (tr, te) in enumerate(folds):
            wtr = make_windows(tr)
            wte = make_windows(te)
            model = make_model(key).fit(wtr)
            m = trainable(wte)
            wv = wte.subset(m)
            t1 = time.perf_counter()
            p = model.predict_proba(wv)
            infer_ms.append(1000 * (time.perf_counter() - t1) / max(1, len(wv)))
            ys.append(wv.y)
            ps.append(p)
            fold_m.append(window_metrics(wv.y, p))
            st = stream_predictions(model, te)
            streams += st
            # rendimiento por grupo (escena o secuencia)
            for g in np.unique(wv.group):
                gm = wv.group == g
                if gm.sum() >= 5:
                    rows_g.append(dict(protocol=name, model=key, group=g, **window_metrics(wv.y[gm], p[gm])))
            if occlusion:
                for (kind, q), (yy, pp) in occ.items():
                    wo = wv if q == 0 else occlude(wv, kind, q, rng)
                    yy.append(wv.y)
                    pp.append(p if q == 0 else model.predict_proba(wo))
        y, p = np.concatenate(ys), np.concatenate(ps)
        wm = window_metrics(y, p)
        std = {f"{k}_std": float(np.nanstd([f[k] for f in fold_m])) for k in MAIN_METRICS}
        rows_w.append(dict(protocol=name, model=key, n_folds=len(folds), **wm, **std,
                           infer_ms_per_window=float(np.mean(infer_ms)),
                           time_s=time.perf_counter() - t0))
        for em in all_event_metrics(streams):
            em = dict(em)
            em.pop("per_seq")
            rows_e.append(dict(protocol=name, model=key, **em))
        if occlusion:
            for (kind, q), (yy, pp) in occ.items():
                om = window_metrics(np.concatenate(yy), np.concatenate(pp))
                rows_o.append(dict(protocol=name, model=key, kind=kind, level=q, f1=om["f1"],
                                   sensitivity=om["sensitivity"], specificity=om["specificity"]))
        log(f"  {MODEL_LABEL[key]:16s} F1={wm['f1']:.3f}  sens={wm['sensitivity']:.3f}  "
            f"esp={wm['specificity']:.3f}  AUC={wm['roc_auc']:.3f}  ({time.perf_counter() - t0:.0f} s)")
    return rows_w, rows_e, rows_g, rows_o, rows_t


# ---------------------------------------------------------------------------
# Privacidad y coste
# ---------------------------------------------------------------------------
def privacy_cost_table(data: dict[str, list[PreparedSeq]]) -> pd.DataFrame:
    rows = []
    for ds, seqs in data.items():
        if not seqs:
            continue
        meta = pd.DataFrame([s.meta for s in seqs])
        fps = float(meta["fps"].median())
        w, h = int(meta["width"].median()), int(meta["height"].median())
        jpeg = float(meta["jpeg_bytes"].median())
        skel = C.N_KP * 3 * 4                       # 17 puntos x (x, y, conf) float32
        rows += [
            dict(dataset=ds, representation="Vídeo RGB sin comprimir", bytes_per_s=w * h * 3 * fps,
                 info="Identidad (cara, ropa), hogar, otras personas, actividad"),
            dict(dataset=ds, representation="Vídeo RGB (JPEG q=90)", bytes_per_s=jpeg * fps,
                 info="Igual que el anterior"),
            dict(dataset=ds, representation="Esqueleto 2D (17 puntos)", bytes_per_s=skel * fps,
                 info="Postura, proporciones corporales y forma de andar (re-identificación posible)"),
            dict(dataset=ds, representation="Probabilidad + eventos", bytes_per_s=4 * fps / C.STRIDE_STREAM,
                 info="Solo hora y tipo de evento"),
        ]
        rows.append(dict(dataset=ds, representation="Tiempo de estimación de pose (ms/frame)",
                         bytes_per_s=np.nan, info=f"{meta['pose_ms'].median():.1f} ms"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
PALETTE = {"rules": "#2a78d6", "rf": "#eb6834", "svm": "#1baf7a", "gru": "#eda100",
           "gru_noaug": "#e87ba4", "stgcn": "#6250d6"}
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _style(ax, title=None, ylabel=None):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def make_figures(out: Path, dfw: pd.DataFrame, dfe: pd.DataFrame, dfo: pd.DataFrame,
                 dfg: pd.DataFrame, fi: pd.DataFrame | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF})
    models = [m for m in ALL_MODELS if m in set(dfw.model)]
    protocols = list(dict.fromkeys(dfw.protocol))

    # 1) F1 por modelo y protocolo (muestra la caída de rendimiento entre datasets)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(protocols) + 2), 3.6))
    bw = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [dfw[(dfw.protocol == p) & (dfw.model == m)].f1.mean() for p in protocols]
        errs = [dfw[(dfw.protocol == p) & (dfw.model == m)].f1_std.mean() for p in protocols]
        x = np.arange(len(protocols)) + (i - (len(models) - 1) / 2) * bw
        ax.bar(x, vals, bw * 0.9, color=PALETTE[m], label=MODEL_LABEL[m], yerr=errs,
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.set_xticks(np.arange(len(protocols)), protocols, color=INK)
    ax.set_ylim(0, 1.05)
    _style(ax, "F1 por ventana (caída vs. actividad cotidiana)", "F1")
    _legend(fig, ax, len(models))
    fig.savefig(fig_dir / "f1_por_protocolo.png", dpi=200)
    plt.close(fig)

    # 2) Falsas alarmas por hora y sensibilidad por evento según el nivel de decisión
    level_lbl = {"raw": "Umbral", "persist": "+ persistencia", "confirm": "+ inactividad"}
    for p in protocols:
        sub = dfe[dfe.protocol == p]
        if sub.empty:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        for ax, col, ttl in [(axs[0], "false_alarms_per_hour", "Falsas alarmas por hora"),
                             (axs[1], "event_sensitivity", "Caídas detectadas (sensibilidad)")]:
            levels = ["raw", "persist", "confirm"]
            for i, m in enumerate(models):
                vals = [sub[(sub.model == m) & (sub.level == lv)][col].mean() for lv in levels]
                x = np.arange(3) + (i - (len(models) - 1) / 2) * bw
                ax.bar(x, vals, bw * 0.9, color=PALETTE[m], label=MODEL_LABEL[m])
            ax.set_xticks(np.arange(3), [level_lbl[lv] for lv in levels], color=INK)
            _style(ax, ttl)
            if col == "event_sensitivity":
                ax.set_ylim(0, 1.05)
        fig.suptitle(f"Evaluación por evento · {p}", color=INK, fontsize=11, x=0.01, ha="left")
        _legend(fig, axs[0], len(models))
        fig.savefig(fig_dir / f"eventos_{_slug(p)}.png", dpi=200)
        plt.close(fig)

    # 3) Robustez a oclusiones
    if not dfo.empty:
        p0 = dfo.protocol.iloc[0]
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for ax, kind, ttl in [(axs[0], "random", "Articulaciones perdidas al azar"),
                              (axs[1], "lower", "Piernas ocultas (fracción final de la ventana)")]:
            for m in models:
                s = dfo[(dfo.protocol == p0) & (dfo.model == m) & (dfo.kind == kind)].sort_values("level")
                if len(s):
                    ax.plot(100 * s.level, s.f1, color=PALETTE[m], lw=2, marker="o", ms=5, label=MODEL_LABEL[m])
            ax.set_xlabel("% ocultado", color=INK2, fontsize=9)
            _style(ax, ttl, "F1" if kind == "random" else None)
            ax.set_ylim(0, 1.05)
        fig.suptitle(f"Oclusiones simuladas · {p0}", color=INK, fontsize=11, x=0.01, ha="left")
        _legend(fig, axs[0], len(models))
        fig.savefig(fig_dir / "oclusiones.png", dpi=200)
        plt.close(fig)

    # 4) Rendimiento por escena (Le2i, leave-one-scene-out)
    sc = dfg[dfg.protocol.str.contains("Le2i") & ~dfg.protocol.str.contains("→")]
    if not sc.empty:
        groups = sorted(sc.group.unique())
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(groups) + 2), 3.4))
        for i, m in enumerate(models):
            vals = [sc[(sc.model == m) & (sc.group == g)].f1.mean() for g in groups]
            x = np.arange(len(groups)) + (i - (len(models) - 1) / 2) * bw
            ax.bar(x, vals, bw * 0.9, color=PALETTE[m], label=MODEL_LABEL[m])
        ax.set_xticks(np.arange(len(groups)), [g.replace("_", " ") for g in groups], color=INK, fontsize=8)
        ax.set_ylim(0, 1.05)
        _style(ax, "F1 por escena (campo de visión distinto en cada una)", "F1")
        _legend(fig, ax, len(models))
        fig.savefig(fig_dir / "f1_por_escena.png", dpi=200)
        plt.close(fig)

    # 5) Importancia de las features (Random Forest)
    if fi is not None and len(fi):
        top = fi.head(10)[::-1]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.barh(top.feature, top.importance, color=PALETTE["rf"], height=0.6)
        _style(ax, "Features más relevantes (Random Forest)")
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.yaxis.grid(False)
        ax.tick_params(axis="y", colors=INK)
        fig.tight_layout()
        fig.savefig(fig_dir / "importancia_features.png", dpi=200)
        plt.close(fig)


def _legend(fig, ax, n):
    """Leyenda bajo la figura para que nunca tape las barras."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(n, 6), frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower()


def _md_table(df: pd.DataFrame, cols: list[str], fmt: str = "{:.3f}") -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                         ("–" if isinstance(v, (float, np.floating)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--quick", action="store_true", help="menos épocas (prueba rápida)")
    ap.add_argument("--synthetic", action="store_true", help="usar datos sintéticos (solo prueba)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-occlusion", action="store_true")
    args = ap.parse_args()

    if args.quick:
        C.DEEP.update(epochs=6, patience=3)
    skel_dir = C.SKELETON_DIR
    out = args.out or C.RESULTS_DIR
    if args.synthetic:
        from .synthetic import generate
        skel_dir = generate(C.DATA_DIR / "synthetic_skeletons", n_per_class=8)
        out = args.out or (C.ROOT / "results_synthetic")
    out.mkdir(parents=True, exist_ok=True)

    models = list(args.models)
    if any(m in models for m in ("gru", "gru_noaug", "stgcn")) and not torch_available():
        print("AVISO: PyTorch no está instalado; se omiten los modelos profundos.")
        models = [m for m in models if m in ("rules", "rf", "svm")]

    log_lines: list[str] = []

    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    data = {ds: load_dataset(ds, skel_dir) for ds in ["urfd", "le2i"]}
    for ds, seqs in data.items():
        log(f"[{ds}] {len(seqs)} secuencias ({sum(s.is_fall for s in seqs)} caídas)")
    if not any(data.values()):
        raise SystemExit("No hay esqueletos. Ejecute antes: python -m src.extract_skeletons")

    protocols = []
    names = {"urfd": "URFD", "le2i": "Le2i"}
    for ds, seqs in data.items():
        if len(seqs) >= 4:
            splits, how = seq_splits(seqs)
            folds = [([seqs[i] for i in tr], [seqs[i] for i in te]) for tr, te in splits]
            protocols.append((f"{names[ds]} (CV)", folds, how))
    if data["urfd"] and data["le2i"]:
        protocols.append(("URFD → Le2i", [(data["urfd"], data["le2i"])], "entre datasets"))
        protocols.append(("Le2i → URFD", [(data["le2i"], data["urfd"])], "entre datasets"))

    W, E, G, O = [], [], [], []
    for i, (name, folds, how) in enumerate(protocols):
        log(f"\n=== {name} · {how} · {len(folds)} particiones ===")
        w, e, g, o, _ = run_protocol(name, folds, models, log, occlusion=(i == 0 and not args.no_occlusion))
        W += w
        E += e
        G += g
        O += o
    dfw, dfe, dfg, dfo = (pd.DataFrame(x) for x in (W, E, G, O))
    dfw.to_csv(out / "window_metrics.csv", index=False)
    dfe.to_csv(out / "event_metrics.csv", index=False)
    dfg.to_csv(out / "per_group_metrics.csv", index=False)
    dfo.to_csv(out / "occlusion.csv", index=False)

    # Importancia de features con todo el primer dataset disponible
    fi = None
    if "rf" in models:
        first = next(s for s in data.values() if s)
        rf = FeatureModel("rf").fit(make_windows(first))
        fi = pd.DataFrame(list(rf.feature_importance().items()), columns=["feature", "importance"])
        fi.to_csv(out / "feature_importance.csv", index=False)

    pc = privacy_cost_table(data)
    pc.to_csv(out / "privacy_cost.csv", index=False)
    make_figures(out, dfw, dfe, dfo, dfg, fi)

    # Resumen en Markdown (tablas listas para el póster)
    dfw2 = dfw.assign(modelo=dfw.model.map(MODEL_LABEL))
    dfe2 = dfe.assign(modelo=dfe.model.map(MODEL_LABEL))
    md = ["# Resultados", "",
          "> Generado automáticamente por `python -m src.experiments`."
          + (" **DATOS SINTÉTICOS: solo prueba del pipeline, sin valor científico.**" if args.synthetic else ""),
          "", "## Métricas por ventana (umbral 0.5)", "",
          _md_table(dfw2, ["protocol", "modelo", "sensitivity", "specificity", "precision", "f1",
                           "roc_auc", "infer_ms_per_window"]),
          "", "## Métricas por evento (simulación en streaming)", "",
          _md_table(dfe2, ["protocol", "modelo", "level", "event_sensitivity", "false_alarms",
                           "false_alarms_per_hour", "latency_s"]),
          "", "## Privacidad y coste por representación", "",
          _md_table(pc, ["dataset", "representation", "bytes_per_s", "info"], fmt="{:,.0f}"),
          "", "## Registro", "", "```", *log_lines, "```"]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (out / "config_used.json").write_text(json.dumps(
        {k: v for k, v in vars(C).items() if k.isupper() and isinstance(v, (int, float, str, list, dict, tuple))},
        indent=2, default=str), encoding="utf-8")
    print(f"\nResultados en {out}  (summary.md, *.csv, figures/)")


if __name__ == "__main__":
    main()
