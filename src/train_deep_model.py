"""Entrenamiento de los modelos temporales profundos (GRU / ST-GCN).

Uso (entrena con todo un dataset y guarda el modelo para el servicio):
    python -m src.train_deep_model --train urfd --arch gru
    python -m src.train_deep_model --train urfd le2i --arch stgcn --no-aug
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

from . import config as C
from .models_deep import augment, build
from .preprocessing import Windows, deep_inputs, load_dataset, make_windows
from .train_baseline import trainable


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeepModel:
    def __init__(self, arch: str = "gru", use_aug: bool = True, verbose: bool = False, seed: int = C.SEED):
        self.arch, self.use_aug, self.verbose, self.seed = arch, use_aug, verbose, seed
        self.name = {"gru": "GRU", "stgcn": "ST-GCN"}[arch] + ("" if use_aug else " (sin aum.)")
        self.device = get_device()
        self.net: nn.Module | None = None
        self.train_time_s = 0.0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _split(self, groups: np.ndarray, y: np.ndarray):
        """Separa un subconjunto de validación por grupos (sin fuga entre secuencias)."""
        rng = np.random.RandomState(self.seed)
        for k in range(20):
            gss = GroupShuffleSplit(n_splits=1, test_size=C.DEEP["val_frac"], random_state=rng.randint(10**6))
            tr, va = next(gss.split(np.zeros(len(y)), y, groups))
            if 0 < y[va].sum() < len(va) and y[tr].sum() > 0:
                return tr, va
        idx = rng.permutation(len(y))  # pocos grupos: partición aleatoria estratificada
        n_va = max(1, int(C.DEEP["val_frac"] * len(y)))
        return idx[n_va:], idx[:n_va]

    def fit(self, w: Windows) -> "DeepModel":
        _set_seed(self.seed)
        m = trainable(w)
        X = deep_inputs(w)[m]
        y = w.y[m].astype(np.float32)
        tr, va = self._split(w.group[m], y)
        Xtr, ytr = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
        Xva, yva = torch.from_numpy(X[va]).to(self.device), y[va]

        self.net = build(self.arch).to(self.device)
        pos = max(1.0, float(ytr.sum()))
        pos_weight = torch.tensor([(len(ytr) - pos) / pos], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(self.net.parameters(), lr=C.DEEP["lr"], weight_decay=C.DEEP["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.DEEP["epochs"])

        best, best_state, wait = -np.inf, None, 0
        bs = C.DEEP["batch_size"]
        t0 = time.perf_counter()
        for ep in range(C.DEEP["epochs"]):
            self.net.train()
            perm = torch.randperm(len(Xtr))
            tot = 0.0
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                xb, yb = Xtr[idx].to(self.device), ytr[idx].to(self.device)
                if self.use_aug:
                    xb = augment(xb)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tot += loss.item() * len(idx)
            sched.step()
            p = self._predict_tensor(Xva)
            score = roc_auc_score(yva, p) if 0 < yva.sum() < len(yva) else -float(
                nn.functional.binary_cross_entropy(torch.tensor(p), torch.tensor(yva)))
            self.history.append(dict(epoch=ep, train_loss=tot / len(perm), val_score=score))
            if self.verbose:
                print(f"    época {ep:02d}  loss {tot / len(perm):.4f}  val {score:.4f}")
            if score > best + 1e-4:
                best, best_state, wait = score, copy.deepcopy(self.net.state_dict()), 0
            else:
                wait += 1
                if wait >= C.DEEP["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.train_time_s = time.perf_counter() - t0
        return self

    @torch.no_grad()
    def _predict_tensor(self, X: torch.Tensor, bs: int = 1024) -> np.ndarray:
        self.net.eval()
        out = [torch.sigmoid(self.net(X[i:i + bs].to(self.device))).cpu() for i in range(0, len(X), bs)]
        return torch.cat(out).numpy() if out else np.zeros(0)

    def predict_proba(self, w: Windows) -> np.ndarray:
        X = deep_inputs(w)
        if len(X) == 0:
            return np.zeros(0)
        return self._predict_tensor(torch.from_numpy(X))

    def save(self, path):
        torch.save({"arch": self.arch, "state": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu")
        m = cls(d["arch"])
        m.net = build(d["arch"]).to(m.device)
        m.net.load_state_dict(d["state"])
        m.net.eval()
        return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["urfd"], choices=["urfd", "le2i"])
    ap.add_argument("--arch", default="gru", choices=["gru", "stgcn"])
    ap.add_argument("--no-aug", action="store_true")
    args = ap.parse_args()
    seqs = [s for ds in args.train for s in load_dataset(ds)]
    if not seqs:
        raise SystemExit("No hay esqueletos. Ejecute antes: python -m src.extract_skeletons")
    w = make_windows(seqs)
    model = DeepModel(args.arch, use_aug=not args.no_aug, verbose=True).fit(w)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.MODELS_DIR / f"{args.arch}_{'+'.join(args.train)}.pt"
    model.save(out)
    print(f"Modelo guardado en {out} (dispositivo: {model.device}, {model.train_time_s:.0f} s)")


if __name__ == "__main__":
    main()
