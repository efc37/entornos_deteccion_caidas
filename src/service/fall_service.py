"""Servicio de detección de caídas para una casa inteligente ("casa que te cuida").

Flujo (máquina de estados):

  MONITORIZANDO --p>=umbral en K ventanas--> POSIBLE_CAIDA
  POSIBLE_CAIDA --sigue en el suelo INACTIVITY_S--> PREGUNTANDO  ("¿Se encuentra bien?")
                --se levanta / se mueve----------> MONITORIZANDO (alarma descartada)
  PREGUNTANDO   --"estoy bien"----------------> MONITORIZANDO (falsa alarma evitada)
                --"ayuda" o sin respuesta------> ALERTA (aviso a contacto de confianza)
  Persona no visible > OUT_OF_VIEW_S -> aviso informativo de campo de visión; si
  desaparece justo después de un posible caída, se pregunta igualmente.

Privacidad por diseño: el servicio solo recibe esqueletos (la imagen se descarta en
el extractor), los mantiene unos segundos en memoria y únicamente registra eventos
(hora, tipo, probabilidad). Nada sale de la casa salvo el aviso final.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np

from .. import config as C
from ..postprocess import confirm_down
from ..preprocessing import Windows


class State(str, Enum):
    MONITORING = "MONITORIZANDO"
    CANDIDATE = "POSIBLE_CAIDA"
    ASKING = "PREGUNTANDO"
    ALERT = "ALERTA"


class SmartHome:
    """Actuadores de la casa (sustituibles por altavoz/app/domótica reales)."""

    def speak(self, text: str):
        print(f"    [altavoz] {text}")

    def notify_contact(self, text: str):
        print(f"    [aviso a contacto de confianza] {text}")

    def info(self, text: str):
        print(f"    [registro] {text}")


@dataclass
class Event:
    t: float
    kind: str
    detail: str = ""
    prob: float | None = None


@dataclass
class FallService:
    model: object                                   # cualquier modelo con predict_proba(Windows)
    home: SmartHome = field(default_factory=SmartHome)
    responder: Callable[[float], tuple[str, float] | None] | None = None  # simula la respuesta
    fps: float = C.TARGET_FPS
    thr: float = C.PROB_THRESHOLD

    def __post_init__(self):
        n = C.WINDOW + int(C.INACTIVITY_S * self.fps) + 4
        self.buf_xy: deque = deque(maxlen=n)
        self.buf_conf: deque = deque(maxlen=n)
        self.state = State.MONITORING
        self.frame = 0
        self.run = 0
        self.cand_frame = None
        self.ask_t = None
        self.pending_answer = None
        self.last_alarm_t = -1e9
        self.unseen = 0
        self.unseen_reported = False
        self.events: list[Event] = []
        self.trace: list[tuple[float, float, str]] = []   # (t, prob, estado) para la figura

    # ------------------------------------------------------------------
    def _log(self, kind: str, detail: str = "", prob: float | None = None):
        e = Event(round(self.frame / self.fps, 2), kind, detail, None if prob is None else round(float(prob), 3))
        self.events.append(e)
        self.home.info(f"t={e.t:6.2f}s {kind} {detail}")

    def _window_prob(self) -> tuple[float, float]:
        xy = np.array(self.buf_xy)[-C.WINDOW:]
        cf = np.array(self.buf_conf)[-C.WINDOW:]
        visible = ~np.isnan(xy[..., 0]).all(axis=1)
        missing = 1.0 - visible.mean()
        w = Windows(xy[None], cf[None], np.zeros(1, np.int8), np.zeros(1, int), np.array(["live"]),
                    np.zeros(1, int), np.array([missing], np.float32))
        return float(self.model.predict_proba(w)[0]), float(missing)

    def push(self, xy: np.ndarray, conf: np.ndarray):
        """Procesa un frame de esqueleto: xy (17,2) con NaN si no visible, conf (17,)."""
        self.buf_xy.append(xy.astype(np.float32))
        self.buf_conf.append(conf.astype(np.float32))
        self.frame += 1
        t = self.frame / self.fps
        visible = np.isfinite(xy).any()
        self.unseen = 0 if visible else self.unseen + 1

        # --- campo de visión -------------------------------------------------
        if self.unseen >= C.OUT_OF_VIEW_S * self.fps and not self.unseen_reported:
            self.unseen_reported = True
            if self.state == State.CANDIDATE:
                self._ask(t, "la persona desapareció del campo de visión tras un posible caída")
            else:
                self._log("FUERA_DE_VISTA", "persona no visible: revisar el campo de visión de la cámara")
        if visible:
            self.unseen_reported = False

        # --- respuesta a la pregunta de confirmación --------------------------
        if self.state == State.ASKING:
            self._handle_answer(t)
            self.trace.append((t, np.nan, self.state.value))
            return

        if self.frame < C.WINDOW or self.frame % C.STRIDE_STREAM:
            self.trace.append((t, np.nan, self.state.value))
            return
        p, missing = self._window_prob()
        self.trace.append((t, p, self.state.value))

        if self.state == State.MONITORING:
            above = p >= self.thr and missing <= C.MAX_MISSING_FRAC
            self.run = self.run + 1 if above else 0
            if self.run >= C.PERSIST_K and t - self.last_alarm_t > C.REFRACTORY_S:
                self.state, self.cand_frame, self.run = State.CANDIDATE, self.frame, 0
                self._log("POSIBLE_CAIDA", "patrón de caída persistente", p)
        elif self.state == State.CANDIDATE:
            waited = self.frame - self.cand_frame
            if waited >= C.INACTIVITY_S * self.fps:
                xy_after = np.array(self.buf_xy)[-waited:]
                if confirm_down(xy_after, self.fps):
                    self._ask(t, "la persona sigue en el suelo")
                else:
                    self._log("DESCARTADA", "la persona se ha levantado o se mueve con normalidad", p)
                    self.state = State.MONITORING
                    self.last_alarm_t = t

    # ------------------------------------------------------------------
    def _ask(self, t: float, reason: str):
        self.state, self.ask_t = State.ASKING, t
        self._log("PREGUNTA", reason)
        self.home.speak("Parece que ha podido caerse. ¿Se encuentra bien? Diga 'estoy bien' o 'ayuda'.")
        self.pending_answer = self.responder(t) if self.responder else None

    def _handle_answer(self, t: float):
        ans = self.pending_answer
        if ans is not None and t - self.ask_t >= ans[1]:
            if ans[0] == "ok":
                self._log("CANCELADA", "la persona confirma que está bien: no se avisa a nadie")
                self.home.speak("De acuerdo. Sigo atento.")
                self._reset(t)
            else:
                self._alert(t, "la persona pide ayuda")
            return
        if t - self.ask_t >= C.CONFIRM_TIMEOUT_S:
            self._alert(t, f"sin respuesta en {C.CONFIRM_TIMEOUT_S:.0f} s")

    def _alert(self, t: float, reason: str):
        self.state = State.ALERT
        self._log("ALERTA", reason)
        self.home.notify_contact(f"Posible caída detectada a las t={t:.0f}s ({reason}).")
        self.home.speak("He avisado a su contacto de confianza.")
        self._reset(t)

    def _reset(self, t: float):
        self.state, self.cand_frame, self.ask_t, self.pending_answer = State.MONITORING, None, None, None
        self.last_alarm_t = t
        self.run = 0

    def save_log(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([e.__dict__ for e in self.events], indent=2, ensure_ascii=False),
                        encoding="utf-8")
