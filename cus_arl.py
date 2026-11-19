"""Exakte Lauflängen-Rechnung für Shewhart-Karte und CUSUM (Beobachtungen N(delta, 1), Ziel 0, sigma = 1 bekannt, unabhängig).

Zähleinheit überall: Beobachtungen bis zum Alarm (die Alarmbeobachtung zählt mit). ARL0 = Lauflänge ohne Änderung, ARL1 = Lauflänge, wenn die
Änderung vor der ersten Beobachtung liegt (nu = 0, zero-state); für beide Karten dieser Demo ist das zugleich das Lorden-Maß.

Zwei unabhängige exakte Wege für den einseitigen CUSUM (S_0 = 0, S_t = max(0, S_(t-1) + x_t - k), Alarm bei S_t >= h):
  * Integralgleichung (Nyström mit Gauß-Legendre-Knoten): `cusum_arl`, `cusum_tune`, live in der App;
  * Brook/Evans-Gitter (Markov-Kette auf Zellen der Weite w = h / (m - 1/2)): `grid_arl`, `grid_matrix`, nur Orakel, Verteilung und Tabellen.
Dazu: zweiseitiger CUSUM als 2D-Kette (nur eingefrorene Tabelle und Test), CUSUM mit Shewhart-Grenze (Gitter, nur Tabelle und Test).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # kleine dichte Lösungen sind mit mehreren BLAS-Threads um den Faktor 100 langsamer

import functools  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402
from scipy.optimize import brentq, minimize_scalar  # noqa: E402
from scipy.special import ndtr, ndtri  # noqa: E402

DEFAULT_NODES = 120  # Knoten der Integralgleichung (Knotenzahl-Kontrolle: Tests)


class NotTunable(ValueError):
    """Die Ziel-ARL0 ist mit diesem k durch keine Schwelle h erreichbar (ARL0 > 1 / Phi(-k) ist nötig)."""


def _phi(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2.0 * math.pi)


# ----------------------------------------------------------------------------- Shewhart-Karte (geschlossen)
def shewhart_c(arl0: float, two: bool = False) -> float:
    """Grenze c mit ARL0 = 1/p: einseitig p = P(x > c) = 1/ARL0, zweiseitig P(|x| > c) = 1/ARL0."""
    if arl0 <= 1:
        raise ValueError("ARL0 muss größer als 1 sein")
    return float(ndtri(1.0 - (0.5 if two else 1.0) / arl0))


def shewhart_arl(c: float, delta: float, two: bool = False) -> float:
    """ARL1 = 1 / P(Alarm je Beobachtung) bei Mittel delta (Alarm bei x > c, zweiseitig |x| > c)."""
    p = float(ndtr(delta - c))
    if two:
        p += float(ndtr(-c - delta))
    return 1.0 / p


def shewhart_arl_at(arl0: float, delta: float, two: bool = False) -> float:
    return shewhart_arl(shewhart_c(arl0, two), delta, two)


# ----------------------------------------------------------------------------- CUSUM einseitig: Integralgleichung
@functools.lru_cache(maxsize=16)
def _gauss(n: int):
    return leggauss(n)


def cusum_arl(k: float, h: float, delta: float, nodes: int = DEFAULT_NODES) -> float:
    """ARL des einseitigen CUSUM (Start S_0 = 0) aus der Integralgleichung (Page; Goel/Wu), Nyström-Verfahren.

    L(u) = 1 + L(0) Phi(k - u - delta) + Integral_0^h L(y) phi(y - u + k - delta) dy; h <= 0 gibt 1 (Alarm schon bei der ersten Beobachtung)."""
    if h <= 0:
        return 1.0
    x, w = _gauss(nodes)
    y = 0.5 * h * (x + 1.0)
    wq = 0.5 * h * w
    u = np.concatenate([[0.0], y])
    a = np.empty((nodes + 1, nodes + 1))
    a[:, 0] = ndtr(k - u - delta)
    a[:, 1:] = _phi(y[None, :] - u[:, None] + k - delta) * wq[None, :]
    return float(np.linalg.solve(np.eye(nodes + 1) - a, np.ones(nodes + 1))[0])


def max_arl0(k: float) -> float:
    """Kleinste Grenze der erreichbaren ARL0 für h -> 0+: 1 / Phi(-k) (der CUSUM wird zur Shewhart-Karte mit Grenze k)."""
    return 1.0 / float(ndtr(-k))


def cusum_tune(k: float, arl0: float, nodes: int = DEFAULT_NODES) -> float:
    """Schwelle h mit einseitiger ARL0 = arl0 (brentq auf der Integralgleichung). Wirft NotTunable, wenn arl0 <= 1 / Phi(-k)."""
    if arl0 <= max_arl0(k):
        raise NotTunable(f"ARL0 = {arl0:g} ist mit k = {k:g} nicht erreichbar (nötig: ARL0 > {max_arl0(k):.1f})")

    def f(h):
        return cusum_arl(k, h, 0.0, nodes) - arl0

    hi = 1.0
    while f(hi) < 0:
        hi *= 2.0
        if hi > 200:
            raise NotTunable("keine Schwelle gefunden")
    return float(brentq(f, 1e-9, hi, xtol=1e-12))


def arl1_tuned(k: float, arl0: float, delta: float, nodes: int = DEFAULT_NODES) -> float:
    """ARL1 des CUSUM (k) bei auf ARL0 abgestimmter Schwelle."""
    return cusum_arl(k, cusum_tune(k, arl0, nodes), delta, nodes)


def crossover(k: float, arl0: float, nodes: int = DEFAULT_NODES, lo: float = 0.3, hi: float = 7.0, step: float = 0.1):
    """Umschlagpunkt delta*: der kleinste Sprung, ab dem die einseitige Shewhart-Karte bei gleicher ARL0 die kleinere ARL1 hat.

    Raster `step` sucht den ersten Vorzeichenwechsel, brentq verfeinert. None, wenn die Shewhart-Karte schon bei `lo` besser ist oder bis `hi` kein Umschlag kommt."""
    h = cusum_tune(k, arl0, nodes)
    c = shewhart_c(arl0)

    def f(d):
        return cusum_arl(k, h, d, nodes) - shewhart_arl(c, d)

    grid = np.arange(lo, hi + 1e-9, step)
    prev = f(grid[0])
    if prev > 0:
        return None
    for a, b in zip(grid[:-1], grid[1:]):
        fb = f(b)
        if fb >= 0:
            return float(brentq(f, a, b, xtol=1e-10))
        prev = fb
    return None


def best_k(delta: float, arl0: float, nodes: int = 100):
    """Bestes k für feste Sprunggröße delta bei ARL0 (jedes k neu abgestimmt): (k*, ARL1*). Suche über k in [0,03; min(1,5 delta; Obergrenze)]."""
    k_cap = min(1.5 * delta, shewhart_c(arl0) - 1e-3, 2.7)

    def g(k):
        return arl1_tuned(k, arl0, delta, nodes)

    r = minimize_scalar(g, bounds=(0.03, k_cap), method="bounded", options={"xatol": 1e-6})
    return float(r.x), float(r.fun)


def lorden_row(delta: float, gamma: float, nodes: int | None = None):
    """Exakte Verzögerung J(gamma) des CUSUM mit k = delta/2 bei ARL0 = gamma und die Näherung ln(gamma)/I mit I = delta^2/2 (Lorden 1971 laut Quellenbericht)."""
    nodes = nodes or (200 if gamma >= 1e4 else DEFAULT_NODES)
    k = delta / 2.0
    h = cusum_tune(k, gamma, nodes)
    j = cusum_arl(k, h, delta, nodes)
    approx = math.log(gamma) / (delta * delta / 2.0)
    return {"gamma": gamma, "h": h, "J": j, "approx": approx, "shewhart": shewhart_arl_at(gamma, delta)}


# ----------------------------------------------------------------------------- CUSUM einseitig: Brook/Evans-Gitter
@functools.lru_cache(maxsize=64)
def grid_matrix(k: float, h: float, delta: float, m: int):
    """Übergangsmatrix R (m x m, ohne Alarm) des einseitigen CUSUM: Zelle 0 = [0, w/2), Zelle j = [(j - 1/2) w, (j + 1/2) w), w = h / (m - 1/2)."""
    w = h / (m - 0.5)
    v = w * np.arange(m)
    upper = (np.arange(m) + 0.5) * w
    cdf = ndtr(upper[None, :] - v[:, None] + k - delta)
    return np.diff(cdf, axis=1, prepend=0.0)


def grid_arl(k: float, h: float, delta: float, m: int) -> float:
    r = grid_matrix(k, h, delta, m)
    return float(np.linalg.solve(np.eye(m) - r, np.ones(m))[0])


def richardson(m1: int, a1: float, m2: int, a2: float) -> float:
    """Extrapolation für einen Fehler proportional zu 1/m^2."""
    return (m2**2 * a2 - m1**2 * a1) / (m2**2 - m1**2)


def grid_arl_richardson(k: float, h: float, delta: float, m1: int = 160, m2: int = 320) -> float:
    return richardson(m1, grid_arl(k, h, delta, m1), m2, grid_arl(k, h, delta, m2))


def length_distribution(r: np.ndarray, tmax_cap: int = 6000, eps: float = 1e-9):
    """Exakte Lauflängenverteilung aus der Kette r (Start in Zelle 0): pmf[t] = P(Lauflänge = t), t = 0..T (pmf[0] = 0), bis P(> T) < eps oder T = tmax_cap."""
    m = r.shape[0]
    p = np.zeros(m)
    p[0] = 1.0
    surv = [1.0]
    while surv[-1] >= eps and len(surv) <= tmax_cap:
        p = p @ r
        surv.append(float(p.sum()))
    surv = np.array(surv)
    pmf = np.concatenate([[0.0], surv[:-1] - surv[1:]])
    return pmf[: len(surv)]


def quantile(pmf: np.ndarray, q: float) -> int:
    """Kleinstes t mit P(Lauflänge <= t) >= q."""
    cdf = np.cumsum(pmf)
    return int(np.searchsorted(cdf, q))


def cdf_at(pmf: np.ndarray, t: int) -> float:
    return float(np.sum(pmf[: t + 1]))


def quasi_stationary_delay(k: float, h: float, delta: float, m: int = 400) -> float:
    """Quasistationäre Verzögerung des CUSUM: Mittel von ARL1(u) unter der quasistationären Verteilung der alarmfreien Kette bei delta = 0 (Perron-Vektor).

    Das ist nicht das Shiryaev-Maß (zyklisch, Stück 3); es ist der Grenzwert der Verzögerung für nu -> unendlich, bedingt auf „kein Alarm davor“."""
    r0 = grid_matrix(k, h, 0.0, m)
    ev, vec = np.linalg.eig(r0.T)
    q = np.abs(vec[:, int(np.argmax(ev.real))].real)
    q = q / q.sum()
    r1 = grid_matrix(k, h, delta, m)
    return float(q @ np.linalg.solve(np.eye(m) - r1, np.ones(m)))


# ----------------------------------------------------------------------------- CUSUM zweiseitig: 2D-Kette
def grid2_arl(k: float, h: float, delta: float, m: int) -> float:
    """Zweiseitiger CUSUM (S+, S-) als Markov-Kette auf m x m Zellen (Zellen wie im einseitigen Gitter, beide Statistiken hängen am selben x)."""
    w = h / (m - 0.5)
    v = w * np.arange(m)
    q = np.arange(1, m + 1)
    rows, cols, data = [], [], []
    for i in range(m):
        t_plus = (q - 0.5) * w - v[i] + k
        for j in range(m):
            s_q = v[j] - k - (q - 0.5) * w
            b = np.sort(np.concatenate([t_plus, s_q]))
            lo_e = np.concatenate([[-np.inf], b])
            hi_e = np.concatenate([b, [np.inf]])
            mid = np.where(np.isfinite(lo_e) & np.isfinite(hi_e), 0.5 * (lo_e + hi_e), np.where(np.isfinite(lo_e), lo_e + 1.0, hi_e - 1.0))
            pr = ndtr(hi_e - delta) - ndtr(lo_e - delta)
            ip = np.searchsorted(t_plus, mid, side="right")
            jp = m - np.searchsorted(s_q[::-1], mid, side="left")
            ok = (ip < m) & (jp < m) & (pr > 0)
            rows.append(np.full(int(ok.sum()), i * m + j))
            cols.append(ip[ok] * m + jp[ok])
            data.append(pr[ok])
    r = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(m * m, m * m))
    sol = spl.spsolve(sp.identity(m * m, format="csc") - r.tocsc(), np.ones(m * m))
    return float(sol[0])


def grid2_arl_richardson(k: float, h: float, delta: float, m1: int = 60, m2: int = 80) -> float:
    return richardson(m1, grid2_arl(k, h, delta, m1), m2, grid2_arl(k, h, delta, m2))


# ----------------------------------------------------------------------------- CUSUM mit Shewhart-Grenze (einseitig)
def hybrid_matrix(k: float, h: float, delta: float, c: float, m: int):
    """Wie `grid_matrix`, aber Alarm auch bei x > c: die Zellmassen werden bei x <= c abgeschnitten."""
    w = h / (m - 0.5)
    v = w * np.arange(m)
    upper = (np.arange(m) + 0.5) * w
    ub = np.minimum(upper[None, :] - v[:, None] + k, c)
    return np.diff(ndtr(ub - delta), axis=1, prepend=0.0)


def hybrid_arl_grid(k: float, h: float, delta: float, c: float, m: int) -> float:
    r = hybrid_matrix(k, h, delta, c, m)
    return float(np.linalg.solve(np.eye(m) - r, np.ones(m))[0])


def hybrid_arl(k: float, h: float, delta: float, c: float, m: int = 320) -> float:
    """ARL des CUSUM mit Shewhart-Grenze, Richardson aus m und 2m."""
    return richardson(m, hybrid_arl_grid(k, h, delta, c, m), 2 * m, hybrid_arl_grid(k, h, delta, c, 2 * m))


def hybrid_tune(k: float, c: float, arl0: float, m: int = 320) -> float:
    """Schwelle h, bei der der CUSUM mit Grenze c die ARL0 trifft. Wirft NotTunable, wenn schon die Grenze allein eine kleinere ARL0 hat."""
    if c < 1e8 and 1.0 / float(ndtr(-c)) <= arl0:
        raise NotTunable(f"die Grenze c = {c:g} allein hat ARL0 = {1.0 / float(ndtr(-c)):.1f}, nicht mehr als {arl0:g}")
    return float(brentq(lambda h: hybrid_arl_grid(k, h, 0.0, c, m) - arl0, 0.5, 12.0, xtol=1e-11))
