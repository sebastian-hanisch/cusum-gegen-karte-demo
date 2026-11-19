"""Szenario: die Reihe mit Sprung (Vehikel A des Plans: ausgedachte Normalreihe mit bekanntem Sprung, kein Betriebsmodell).

x_t ~ N(0, 1) für t <= nu, N(delta, 1) danach; Ziel 0 und sigma = 1 sind bekannt, die Beobachtung ist standardisiert. Zufall nur aus cus_rng (SplitMix64)."""

from dataclasses import dataclass

import cus_chart as K
import cus_constants as C


@dataclass(frozen=True)
class Series:
    xs: tuple
    delta: float
    nu: int
    c: float
    k: float
    h: float
    s_path: tuple
    t_shewhart: int | None   # erste Beobachtung mit x > c (kann vor dem Sprung liegen)
    t_cusum: int | None      # erste Beobachtung mit S >= h


def make_series(seed: int, delta: float, nu: int, c: float, k: float, h: float, n: int = C.SERIES_LENGTH) -> Series:
    xs = K.draw_series(seed, n, delta, nu)
    path = K.cusum_path(xs, k)
    return Series(tuple(xs), delta, nu, c, k, h, tuple(path), K.shewhart_alarm(xs, c), K.first_index(path, lambda s: s >= h))


def delay(t_alarm: int | None, nu: int) -> int | None:
    """Verzögerung T - nu (Alarmbeobachtung zählt mit); None bei keinem Alarm oder Alarm vor dem Sprung (Fehlalarm)."""
    if t_alarm is None or t_alarm <= nu:
        return None
    return t_alarm - nu
