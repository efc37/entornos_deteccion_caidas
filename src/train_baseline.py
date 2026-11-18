"""Métodos base: reglas por umbrales, Random Forest y SVM sobre features diseñadas.

Todos los modelos (también los profundos) comparten la interfaz
    model.fit(windows) / model.predict_proba(windows) -> prob. de caída por ventana

Uso (entrena con todo un dataset y guarda el modelo para el servicio):
    python -m src.train_baseline --train urfd --model rf
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C
from .baseline_features import FEATURE_NAMES, features, rule_score
from .preprocessing import Windows, load_dataset, make_windows


def trainable(w: Windows) -> np.ndarray:
    """Ventanas utilizables: etiqueta no ignorada y persona visible."""
    return (w.y != C.WIN_IGNORE) & (w.missing <= C.MAX_MISSING_FRAC)


class FeatureModel:
    """kind = 'rules' | 'rf' | 'svm'."""

    def __init__(self, kind: str = "rf"):
        self.kind = kind
        self.name = {"rules": "Reglas", "rf": "Random Forest", "svm": "SVM"}[kind]
        self.clf = None
        self.train_time_s = 0.0

    def fit(self, w: Windows) -> "FeatureModel":
        if self.kind == "rules":
            return self
        m = trainable(w)
        F, y = features(w)[m], w.y[m].astype(int)
        if self.kind == "rf":
            self.clf = RandomForestClassifier(**C.RF_PARAMS)
        else:
            self.clf = make_pipeline(StandardScaler(), SVC(**C.SVM_PARAMS))
        t0 = time.perf_counter()
        self.clf.fit(F, y)
        self.train_time_s = time.perf_counter() - t0
        return self

    def predict_proba(self, w: Windows) -> np.ndarray:
        F = features(w)
        if len(F) == 0:
            return np.zeros(0)
        if self.kind == "rules":
            return rule_score(F)
        if self.kind == "svm":   # margen de la SVM -> (0,1); 0.5 = frontera de decisión
            return 1.0 / (1.0 + np.exp(-self.clf.decision_function(F)))
        return self.clf.predict_proba(F)[:, 1]

    def feature_importance(self) -> dict | None:
        if self.kind != "rf" or self.clf is None:
            return None
        return dict(sorted(zip(FEATURE_NAMES, self.clf.feature_importances_), key=lambda t: -t[1]))

    def save(self, path):
        joblib.dump({"kind": self.kind, "clf": self.clf}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        m = cls(d["kind"])
        m.clf = d["clf"]
        return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["urfd"], choices=["urfd", "le2i"])
    ap.add_argument("--model", default="rf", choices=["rules", "rf", "svm"])
    args = ap.parse_args()
    seqs = [s for ds in args.train for s in load_dataset(ds)]
    if not seqs:
        raise SystemExit("No hay esqueletos. Ejecute antes: python -m src.extract_skeletons")
    w = make_windows(seqs)
    model = FeatureModel(args.model).fit(w)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.MODELS_DIR / f"{args.model}_{'+'.join(args.train)}.joblib"
    model.save(out)
    print(f"Modelo guardado en {out} ({trainable(w).sum()} ventanas de entrenamiento)")
    if model.feature_importance():
        print("Features más importantes:")
        for k, v in list(model.feature_importance().items())[:8]:
            print(f"  {k:22s} {v:.3f}")


if __name__ == "__main__":
    main()
