"""Pruebas del pipeline con datos sintéticos (no necesitan los datasets reales).

Ejecutar desde la raíz:  python -m pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest

from src import config as C
from src.baseline_features import FEATURE_NAMES, features
from src.datasets import Sequence, _parse_le2i_annotation
from src.evaluation import all_event_metrics, stream_predictions, window_metrics
from src.preprocessing import clean, deep_inputs, load_dataset, make_windows, resample, window_label
from src.service.fall_service import FallService, SmartHome
from src.synthetic import generate
from src.train_baseline import FeatureModel, trainable


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    d = generate(tmp_path_factory.mktemp("syn"), n_per_class=5)
    return load_dataset("urfd", d), load_dataset("le2i", d)


def test_urfd_labels():
    s = Sequence("urfd/fall-01", "urfd", "fall-01", None, 30.0, True,
                 urfd_labels={i: (-1 if i < 10 else 0 if i < 15 else 1) for i in range(30)})
    y = s.frame_labels(30)
    assert (y[:10] == 0).all() and (y[10:15] == 1).all() and (y[15:] == 2).all()
    assert (Sequence("urfd/adl-01", "urfd", "adl-01", None, 30.0, False).frame_labels(5) == 0).all()


def test_le2i_labels(tmp_path):
    f = tmp_path / "video (1).txt"
    f.write_text("48\n80\n1,1,10,10,50,50\n")
    assert _parse_le2i_annotation(f) == (48, 80)
    s = Sequence("le2i/x", "le2i", "Home_01", None, 25.0, True, fall_span=(47, 79))
    y = s.frame_labels(100)
    assert y[46] == 0 and y[47] == 1 and y[79] == 1 and y[80] == 2


def test_clean_and_resample():
    kps = np.random.rand(30, 17, 3).astype(np.float32) * 100
    kps[..., 2] = 0.9
    kps[5:8, :, 2] = 0.0           # hueco corto -> se interpola
    xy, conf = clean(kps)
    assert np.isfinite(xy[5:8]).all()
    xy2, c2, l2 = resample(xy, conf, np.zeros(30, np.int8), 30.0, 25.0)
    assert abs(len(xy2) - 25) <= 1


def test_window_label():
    assert window_label(np.array([0] * 32), 0) == C.WIN_NEG
    assert window_label(np.array([0] * 20 + [1] * 12), 12) == C.WIN_POS
    assert window_label(np.array([0] * 30 + [1] * 2), 12) == C.WIN_IGNORE
    assert window_label(np.array([2] * 32), 12) == C.WIN_IGNORE


def test_features_and_inputs(syn):
    w = make_windows(syn[0])
    F = features(w)
    assert F.shape == (len(w), len(FEATURE_NAMES)) and np.isfinite(F).all()
    X = deep_inputs(w)
    assert X.shape[1:] == (C.WINDOW, C.N_KP, 5) and np.isfinite(X).all()
    # la inclinación final del tronco es mayor en ventanas de caída
    i = FEATURE_NAMES.index("torso_angle_last")
    assert F[w.y == 1, i].mean() > F[w.y == 0, i].mean()


@pytest.mark.parametrize("kind", ["rules", "rf", "svm"])
def test_models_learn(syn, kind):
    urfd, le2i = syn
    m = FeatureModel(kind).fit(make_windows(urfd))
    w = make_windows(le2i)
    w = w.subset(trainable(w))
    met = window_metrics(w.y, m.predict_proba(w))
    assert met["roc_auc"] > 0.8
    ev = all_event_metrics(stream_predictions(m, le2i))
    assert [e["level"] for e in ev] == ["raw", "persist", "confirm"]
    # la persistencia nunca aumenta las falsas alarmas
    assert ev[1]["false_alarms"] <= ev[0]["false_alarms"]


def test_service_flow(syn):
    urfd, le2i = syn
    model = FeatureModel("rf").fit(make_windows(urfd))
    fall = next(s for s in le2i if s.is_fall)
    for resp, final in [(None, "ALERTA"), (("ok", 2.0), "CANCELADA"), (("help", 1.0), "ALERTA")]:
        home = SmartHome()
        home.info = lambda *_: None
        home.speak = lambda *_: None
        home.notify_contact = lambda *_: None
        svc = FallService(model, home, responder=lambda t, r=resp: r)
        for xy, cf in zip(fall.xy, fall.conf):
            svc.push(xy, cf)
        for _ in range(int((C.CONFIRM_TIMEOUT_S + 2) * C.TARGET_FPS)):
            svc.push(fall.xy[-1], fall.conf[-1])
        kinds = [e.kind for e in svc.events]
        assert "PREGUNTA" in kinds and kinds[-1] == final


def test_out_of_view_event(syn):
    model = FeatureModel("rules")
    home = SmartHome()
    home.info = home.speak = home.notify_contact = lambda *_: None
    svc = FallService(model, home)
    nan = np.full((17, 2), np.nan, np.float32)
    for _ in range(int((C.OUT_OF_VIEW_S + 1) * C.TARGET_FPS)):
        svc.push(nan, np.zeros(17, np.float32))
    assert any(e.kind == "FUERA_DE_VISTA" for e in svc.events)


def test_deep_model_if_torch(syn):
    torch = pytest.importorskip("torch")
    from src.models_deep import augment, build
    from src.train_deep_model import DeepModel
    x = torch.randn(4, C.WINDOW, C.N_KP, 5)
    for arch in ["gru", "stgcn"]:
        assert build(arch)(x).shape == (4,)
    assert augment(x).shape == x.shape
    C.DEEP.update(epochs=2)
    m = DeepModel("gru").fit(make_windows(syn[0]))
    p = m.predict_proba(make_windows(syn[1]))
    assert p.min() >= 0 and p.max() <= 1
