"""Modelos temporales sobre secuencias de esqueleto (PyTorch).

* GRUClassifier: proyección por frame + GRU unidireccional (causal, apta para
  streaming en el borde) + pooling (último estado y media) + MLP.
* STGCN: red convolucional de grafos espacio-temporal ligera (Yan et al., 2018;
  aplicada a caídas por Keskes & Noumeir, 2021) sobre el grafo COCO-17.

Entrada: tensor (B, T, V=17, C=5) con x, y, confianza, dx, dy normalizados.
Salida: logit de "caída" (B,).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import config as C


class GRUClassifier(nn.Module):
    def __init__(self, in_ch: int = 5, n_kp: int = C.N_KP, hidden: int = 96, layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_ch * n_kp, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                  nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, V, Ch = x.shape
        h = self.proj(x.reshape(B, T, V * Ch))
        out, _ = self.gru(h)
        z = torch.cat([out[:, -1], out.mean(dim=1)], dim=-1)
        return self.head(z).squeeze(-1)


def normalized_adjacency(n: int = C.N_KP, edges=C.SKELETON_EDGES) -> np.ndarray:
    A = np.eye(n, dtype=np.float32)
    for a, b in edges:
        A[a, b] = A[b, a] = 1.0
    d = A.sum(1)
    Dm = np.diag(1.0 / np.sqrt(d))
    return (Dm @ A @ Dm).astype(np.float32)


class STGCNBlock(nn.Module):
    def __init__(self, cin: int, cout: int, A: torch.Tensor, stride: int = 1, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("A", A)
        self.edge_w = nn.Parameter(torch.ones_like(A))          # importancia aprendida de aristas
        self.gcn = nn.Conv2d(cin, cout, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(cout), nn.Dropout(dropout),
        )
        if cin == cout and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(cin, cout, kernel_size=1, stride=(stride, 1)),
                                     nn.BatchNorm2d(cout))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, C, T, V)
        a = self.A * self.edge_w
        g = torch.einsum("bctv,vw->bctw", x, a)
        return self.relu(self.tcn(self.gcn(g)) + self.res(x))


class STGCN(nn.Module):
    def __init__(self, in_ch: int = 5, channels=(32, 64, 64), dropout: float = 0.3):
        super().__init__()
        A = torch.from_numpy(normalized_adjacency())
        self.bn_in = nn.BatchNorm1d(in_ch * C.N_KP)
        layers, cin = [], in_ch
        for i, cout in enumerate(channels):
            layers.append(STGCNBlock(cin, cout, A, stride=2 if i == 1 else 1))
            cin = cout
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, T, V, C)
        B, T, V, Ch = x.shape
        z = x.permute(0, 3, 2, 1).reshape(B, Ch * V, T)      # normalización por canal y articulación
        z = self.bn_in(z).reshape(B, Ch, V, T).permute(0, 1, 3, 2)  # (B, C, T, V)
        z = self.blocks(z)
        return self.head(z.mean(dim=(2, 3))).squeeze(-1)


def build(arch: str) -> nn.Module:
    if arch == "gru":
        return GRUClassifier(hidden=C.DEEP["hidden"], layers=C.DEEP["layers"], dropout=C.DEEP["dropout"])
    if arch == "stgcn":
        return STGCN(dropout=C.DEEP["dropout"])
    raise ValueError(arch)


# ---------------------------------------------------------------------------
# Aumento de datos (robustez a punto de vista, escala y oclusiones)
# ---------------------------------------------------------------------------
_FLIP_IDX = list(range(C.N_KP))
for a, b in C.FLIP_PAIRS:
    _FLIP_IDX[a], _FLIP_IDX[b] = b, a


def augment(x: torch.Tensor, cfg: dict = C.AUG) -> torch.Tensor:
    """x (B,T,V,5) = x, y, conf, dx, dy. Devuelve una copia aumentada."""
    x = x.clone()
    B, T, V, _ = x.shape
    dev = x.device
    # 1) volteo horizontal (intercambia izquierda/derecha)
    flip = torch.rand(B, device=dev) < cfg["p_flip"]
    if flip.any():
        xf = x[flip][:, :, _FLIP_IDX]
        xf[..., 0] *= -1
        xf[..., 3] *= -1
        x[flip] = xf
    # 2) rotación en el plano de la imagen y 3) cambio de escala
    ang = (torch.rand(B, device=dev) * 2 - 1) * np.radians(cfg["max_rot_deg"])
    s = torch.empty(B, device=dev).uniform_(*cfg["scale"])
    c, si = torch.cos(ang)[:, None, None] * s[:, None, None], torch.sin(ang)[:, None, None] * s[:, None, None]
    for i, j in [(0, 1), (3, 4)]:
        px, py = x[..., i].clone(), x[..., j].clone()
        x[..., i] = c * px - si * py
        x[..., j] = si * px + c * py
    # 4) pérdida aleatoria de articulaciones (fallos del estimador de pose)
    drop = torch.rand(B, T, V, device=dev) < cfg["p_joint_drop"]
    x[drop] = 0.0
    # 5) oclusión de la parte inferior del cuerpo durante un tramo (mobiliario)
    occ = torch.rand(B, device=dev) < cfg["p_lower_occlusion"]
    for b in torch.where(occ)[0].tolist():
        t0 = int(torch.randint(0, T // 2, (1,)))
        t1 = t0 + int(torch.randint(T // 4, T // 2 + 1, (1,)))
        x[b, t0:t1][:, C.LOWER_BODY] = 0.0
    return x
