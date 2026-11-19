"""Orakel: die ARL des einseitigen CUSUM auf unabhängigen Wegen (ORACLE-PLAYBOOK §5).

Wege: (1) Integralgleichung nach Nyström (cus_arl.cusum_arl, live in der App), (2) Brook/Evans-Gitter (cus_arl.grid_arl, Richardson aus zwei Zellzahlen),
(3) die Gitter-Zellmassen noch einmal durch numerische Integration der Normaldichte (scipy.integrate.quad statt ndtr), (4) für die Shewhart-Karte die geschlossene Formel 1/p.
Die Zahlen (335,3675776 usw.) stammen aus der Integralgleichung mit 10 stabilen Stellen ab 20 Knoten (Plan, Vormessung V1 2a) und gelten hier als Handwerte.
Modell: unabhängige Beobachtungen N(delta, 1), Ziel 0, sigma = 1; Zähleinheit Beobachtungen bis zum Alarm."""

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

import cus_arl as A


@pytest.mark.parametrize(("h", "delta", "erwartet"), [(4.0, 0.0, 335.3675776), (4.0, 1.0, 8.3832021), (5.0, 0.0, 930.8870121), (5.0, 1.0, 10.3759753)])
def test_integralgleichung_gegen_handwerte(h, delta, erwartet):
    assert A.cusum_arl(0.5, h, delta) == pytest.approx(erwartet, rel=1e-8)


def test_knotenzahl_kontrolle_zehn_stellen_stabil():
    werte = [A.cusum_arl(0.5, 4.0, 0.0, nodes=n) for n in (40, 80, 120, 200)]
    assert max(werte) - min(werte) < 1e-7 * werte[0]


@pytest.mark.parametrize(("h", "delta", "rel"), [(4.0, 0.0, 2e-3 / 335), (4.0, 1.0, 1e-5), (5.0, 0.0, 2e-3 / 930), (5.0, 1.0, 1e-5), (4.0954, 0.5, 1e-5), (3.0, 2.0, 1e-5)])
def test_gitter_richardson_gegen_integralgleichung(h, delta, rel):
    ref = A.cusum_arl(0.5, h, delta)
    assert A.grid_arl_richardson(0.5, h, delta, 160, 320) == pytest.approx(ref, rel=max(rel, 1e-6))


def test_gitterfehler_wird_mit_der_zellzahl_viermal_kleiner():
    """Fehler proportional zu 1/m^2: Verdopplung von m teilt den Fehler durch 4 (Faktor 4,01 in der Vormessung)."""
    ref = A.cusum_arl(0.5, 4.0, 0.0)
    e160 = ref - A.grid_arl(0.5, 4.0, 0.0, 160)
    e320 = ref - A.grid_arl(0.5, 4.0, 0.0, 320)
    assert 3.5 < e160 / e320 < 4.5
    assert abs(A.grid_arl(0.5, 4.0, 0.0, 160) - 335.3335) < 5e-4 and abs(A.grid_arl(0.5, 4.0, 0.0, 320) - 335.3591) < 5e-4


def test_zufallsinstanzen_gitter_gegen_integralgleichung():
    rng = np.random.default_rng(20261118)
    for _ in range(30):
        k = float(rng.uniform(0.1, 1.2))
        h = float(rng.uniform(1.0, 6.0))
        delta = float(rng.uniform(-0.5, 2.0))
        a = A.cusum_arl(k, h, delta)
        g = A.grid_arl_richardson(k, h, delta, 200, 400)
        assert g == pytest.approx(a, rel=1e-4), (k, h, delta)


def test_zellmassen_des_gitters_gegen_quadratur_der_normaldichte():
    """Die Übergangsmatrix des Gitters aus ndtr gegen eine Quadratur der Dichte je Zelle (anderer Rechenweg für dieselben Zahlen)."""
    k, h, delta, m = 0.5, 3.0, 0.3, 24
    r = A.grid_matrix(k, h, delta, m)
    w = h / (m - 0.5)
    for i in (0, 1, 7, m - 1):
        v = w * i
        for j in range(m):
            lo = -np.inf if j == 0 else (j - 0.5) * w
            hi = (j + 0.5) * w
            val, _ = integrate.quad(lambda y: norm.pdf(y - v + k - delta), lo, hi)
            assert r[i, j] == pytest.approx(val, abs=1e-12), (i, j)


def test_shewhart_geschlossen_gegen_direkte_summe_der_geometrischen_reihe():
    for c, delta in ((2.7818, 0.0), (2.7818, 1.0), (3.0, 4.0), (1.5, -0.5)):
        p = 1 - norm.cdf(c - delta)
        reihe = sum(t * (1 - p) ** (t - 1) * p for t in range(1, 20000))
        assert A.shewhart_arl(c, delta) == pytest.approx(reihe, rel=1e-9)


def test_shewhart_bekannte_zahlen_3_sigma_zweiseitig():
    assert 1 / (2 * (1 - norm.cdf(3.0))) == pytest.approx(370.3983, abs=1e-3)
    assert A.shewhart_arl(3.0, 0.0, two=True) == pytest.approx(370.3983, abs=1e-3)
    assert A.shewhart_arl(3.0, 1.0, two=True) == pytest.approx(43.8947, abs=1e-3)
    assert A.shewhart_c(370.3983473449592, two=True) == pytest.approx(3.0, abs=1e-9)


def test_abgestimmte_schwellen_und_namensfalle_h_4_77():
    """h = 4,0954 gehört einseitig zu ARL0 = 370; h = 4,77 ohne Zusatz ist nur zweiseitig richtig (einseitig gibt es 737,12)."""
    assert A.cusum_tune(0.5, 370) == pytest.approx(4.0954, abs=5e-5)
    assert A.cusum_tune(0.5, 370.398) == pytest.approx(4.0965, abs=5e-5)
    assert A.cusum_arl(0.5, 4.0954, 0.0) == pytest.approx(370, abs=0.02)
    assert A.cusum_arl(0.5, 4.77, 0.0) == pytest.approx(737.12, abs=0.01)
    assert A.shewhart_c(370) == pytest.approx(2.7818, abs=5e-5)
    assert A.shewhart_arl(A.shewhart_c(370), 0.0) == pytest.approx(370, rel=1e-12)


def test_abstimmung_trifft_die_ziel_arl0():
    for k in (0.25, 0.5, 1.0, 1.5):
        for g in (100, 370, 1000):
            assert A.cusum_arl(k, A.cusum_tune(k, g), 0.0) == pytest.approx(g, rel=1e-8)


def test_tabelle_der_schwellen_nach_k():
    """V1 2d: h für ARL0 = 370, k = 0,125 ... 2 (einseitig)."""
    erwartet = {0.125: 9.7441, 0.25: 6.7076, 0.5: 4.0954, 0.75: 2.8820, 1.0: 2.1754, 1.5: 1.3586, 2.0: 0.7929}
    for k, h in erwartet.items():
        assert A.cusum_tune(k, 370) == pytest.approx(h, abs=6e-5), k


def test_arl_steigt_in_h_und_faellt_im_sprung():
    hs = [2.0, 3.0, 4.0, 5.0, 6.0]
    a0 = [A.cusum_arl(0.5, h, 0.0) for h in hs]
    assert all(x < y for x, y in zip(a0, a0[1:]))
    ds = [-0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
    a1 = [A.cusum_arl(0.5, 4.0, d) for d in ds]
    assert all(x > y for x, y in zip(a1, a1[1:]))
    hh = [A.cusum_tune(0.5, g) for g in (50, 100, 200, 370, 1000, 5000)]
    assert all(x < y for x, y in zip(hh, hh[1:]))
