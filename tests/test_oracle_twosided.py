"""Orakel: der zweiseitige CUSUM als 2D-Kette (cus_arl.grid2_arl, Richardson aus m = 60 und 80) auf anderen Wegen.

Wege: die Hälfte der einseitigen ARL bei delta = 0 (Beobachtung, kein Beweis für stetige Normalwerte), eine eigene Kette für auf ein Gitter der Weite w quantisierte
Beobachtungen (hier gilt die Hälftenregel auf Rundungsfehler genau), die Spiegelung delta -> -delta, die Simulation (SplitMix64, Band von 4 Standardfehlern) und die
Handwerte der Vormessung (V11 B). Modell: unabhängige Beobachtungen N(delta, 1), Ziel 0, sigma = 1; Zähleinheit Beobachtungen bis zum Alarm."""

import math

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.special import ndtr

import cus_arl as A
import cus_chart as K
from cus_rng import SplitMix64


def test_zweiseitig_bei_delta_null_ist_die_haelfte_der_einseitigen_arl():
    for h in (4.0, 4.7738):
        zwei = A.grid2_arl_richardson(0.5, h, 0.0)
        ein = A.cusum_arl(0.5, h, 0.0)
        assert zwei == pytest.approx(ein / 2, rel=1e-4), h
    # Handwerte der Vormessung: 167,6844 (h = 4) und 369,9898 (h = 4,7738); die Hälfte der einseitigen ARL ist 167,6838 und 369,9873
    assert A.grid2_arl_richardson(0.5, 4.0, 0.0) == pytest.approx(167.6844, abs=2e-4)
    assert A.grid2_arl_richardson(0.5, 4.7738, 0.0) == pytest.approx(369.9898, abs=2e-4)


def _quantisierte_kette(w, jk, m):
    """Beobachtung auf das Gitter w quantisiert (symmetrisch zu 0), k = jk w, h = m w: ARL einseitig und zweiseitig bei delta = 0 (Zustände in Einheiten von w)."""
    jmax = int(6 / w)
    xs = np.arange(-jmax, jmax + 1)
    kanten = np.concatenate([[-np.inf], (xs[:-1] + xs[1:]) / 2 * w, [np.inf]])
    pr = np.diff(ndtr(kanten))
    a = np.zeros((m, m))
    for s in range(m):
        for x, px in zip(xs, pr):
            t = max(0, s + x - jk)
            if t < m:
                a[s, t] += px
    l1 = np.linalg.solve(np.eye(m) - a, np.ones(m))[0]
    mx = sp.lil_matrix((m * m, m * m))
    for s in range(m):
        for u in range(m):
            for x, px in zip(xs, pr):
                s2, u2 = max(0, s + x - jk), max(0, u - x - jk)
                if s2 < m and u2 < m:
                    mx[s * m + u, s2 * m + u2] += px
    l2 = spl.spsolve(sp.identity(m * m, format="csc") - mx.tocsc(), np.ones(m * m))[0]
    return float(l1), float(l2)


@pytest.mark.parametrize(("w", "jk", "m"), [(0.5, 1, 6), (0.25, 2, 12)])
def test_haelftenregel_auf_einem_beobachtungsgitter_gilt_auf_rundungsfehler(w, jk, m):
    ein, zwei = _quantisierte_kette(w, jk, m)
    assert zwei == pytest.approx(ein / 2, rel=1e-11)


def test_spiegelung_delta_nach_minus_delta():
    for delta in (0.5, 1.0, 2.0):
        a = A.grid2_arl(0.5, 3.0, delta, 24)
        b = A.grid2_arl(0.5, 3.0, -delta, 24)
        assert a == pytest.approx(b, rel=1e-9)


def test_zweiseitig_gegen_einseitig_bei_gleichem_h_handwerte_der_vormessung():
    """δ = 0,5: 35,2534 gegen einseitig 35,2716; δ = 1: beide 9,9246 (der zweite Zweig bringt bei diesen Sprüngen kaum etwas)."""
    assert A.grid2_arl_richardson(0.5, 4.7738, 0.5) == pytest.approx(35.2534, abs=2e-4)
    assert A.cusum_arl(0.5, 4.7738, 0.5) == pytest.approx(35.2716, abs=2e-4)
    assert A.grid2_arl_richardson(0.5, 4.7738, 1.0) == pytest.approx(9.9246, abs=2e-4)
    assert A.cusum_arl(0.5, 4.7738, 1.0) == pytest.approx(9.9246, abs=2e-4)


def test_simulation_des_zweiseitigen_cusum_im_band_der_kette():
    """δ = 1, k = 0,5, h = 4,7738: exakt (2D-Kette) 9,9246; Simulation mit SplitMix64 und eigener Zweig-Rekursion im Test."""
    rng = SplitMix64(31415)
    n = 6000
    laengen = []
    for _ in range(n):
        sp_, sm_ = 0.0, 0.0
        t = 0
        while True:
            t += 1
            x = K.normal_from(rng) + 1.0
            sp_ = max(0.0, sp_ + x - 0.5)
            sm_ = max(0.0, sm_ - x - 0.5)
            if sp_ >= 4.7738 or sm_ >= 4.7738:
                break
        laengen.append(t)
    m, se = K.mean_se(laengen)
    assert abs(m - 9.9246) <= 4 * se, (m, se)


def test_zweiseitige_shewhart_karte_geschlossen_und_gegen_simulation():
    c = A.shewhart_c(370, two=True)
    assert c == pytest.approx(2.9997, abs=5e-5)
    assert A.shewhart_arl(c, 0.0, two=True) == pytest.approx(370, rel=1e-12)
    rng = SplitMix64(2718)
    n = 6000
    laengen = []
    for _ in range(n):
        t = 0
        while True:
            t += 1
            if abs(K.normal_from(rng) + 1.0) > c:
                break
        laengen.append(t)
    m, se = K.mean_se(laengen)
    assert abs(m - A.shewhart_arl(c, 1.0, two=True)) <= 4 * se


def test_umschlag_zweiseitig_aus_der_kette_liegt_hinter_dem_einseitigen():
    """Zweiseitig kippt es später (≈ 2,61) als einseitig (2,4375): Differenz CUSUM − Shewhart wechselt zwischen 2,6 und 2,7 das Vorzeichen (V11 B: −0,0237 und +0,1492)."""
    h = 4.7738
    c = A.shewhart_c(A.grid2_arl_richardson(0.5, h, 0.0), two=True)
    f = lambda d: A.grid2_arl_richardson(0.5, h, d) - A.shewhart_arl(c, d, two=True)  # noqa: E731
    assert f(2.6) == pytest.approx(-0.0237, abs=2e-3) and f(2.7) == pytest.approx(0.1492, abs=2e-3)
    assert f(2.6) < 0 < f(2.7)
    assert math.isclose(A.crossover(0.5, 370), 2.4375, abs_tol=5e-4)
