"""Orakel: Umschlagpunkt, bestes k und die Näherung ln(gamma)/I auf anderen Rechenwegen (ORACLE-PLAYBOOK §5).

* Umschlag delta*: brentq-Nullstelle (cus_arl.crossover, 120 Knoten) gegen den Vorzeichenwechsel auf einem feinen Raster (0,002) mit 200 Knoten.
* Bestes k: bounded-Minimierung (cus_arl.best_k) gegen eine Rastersuche über k (Schrittweite 0,005) und gegen die Gitter-Rechnung am Optimum.
* J(gamma): doppelte Knotenzahl und das Gitter (Richardson).
Modell: unabhängige Beobachtungen N(delta, 1), Ziel 0, sigma = 1, einseitig, zero-state (Lorden-Maß); Zähleinheit Beobachtungen bis zum Alarm."""

import math

import numpy as np
import pytest

import cus_arl as A


def _umschlag_ueber_raster(k, arl0, nodes=200, lo=1.5, hi=4.5, step=0.002):
    h = A.cusum_tune(k, arl0, nodes)
    c = A.shewhart_c(arl0)
    grid = np.arange(lo, hi, step)
    f = np.array([A.cusum_arl(k, h, d, nodes) - A.shewhart_arl(c, d) for d in grid])
    i = int(np.argmax(f >= 0))
    assert i > 0 and f[i - 1] < 0
    return float(grid[i - 1] + step * (-f[i - 1]) / (f[i] - f[i - 1]))   # lineare Interpolation im letzten Rasterintervall


@pytest.mark.parametrize(("k", "arl0"), [(0.5, 370), (0.25, 100), (1.0, 1000)])
def test_umschlag_brentq_gegen_feines_raster(k, arl0):
    assert A.crossover(k, arl0) == pytest.approx(_umschlag_ueber_raster(k, arl0), abs=2e-4)


def test_umschlag_am_umschlagpunkt_gitter_gegen_integralgleichung():
    """Am Umschlagpunkt k = 0,5, ARL0 = 370 gleichen sich beide Rechenwege auf 6 Stellen (Plan V11 A: 2,737602 Integralgleichung, Gitter 2,737612 / 605 / 603 bei m = 160 / 320 / 640)."""
    ds = A.crossover(0.5, 370)
    h = A.cusum_tune(0.5, 370)
    ref = A.cusum_arl(0.5, h, ds)
    assert ref == pytest.approx(2.737602, abs=2e-6)
    assert A.grid_arl(0.5, h, ds, 640) == pytest.approx(ref, abs=5e-5)
    assert ref == pytest.approx(A.shewhart_arl(A.shewhart_c(370), ds), rel=1e-9)   # am Umschlag sind beide Verzögerungen gleich


def test_umschlag_haengt_an_arl0_k_und_ist_nicht_der_eine_wert_2_45():
    """Der Umschlag steigt mit ARL0 und mit k (15 Zellen), langsamer als k; es gibt keinen Umschlag „des CUSUM“."""
    tab = {(g, k): A.crossover(k, g) for g in (100, 370, 1000) for k in (0.25, 0.5, 0.75, 1.0, 1.5)}
    for g in (100, 370, 1000):
        zeile = [tab[(g, k)] for k in (0.25, 0.5, 0.75, 1.0, 1.5)]
        assert all(a < b for a, b in zip(zeile, zeile[1:]))
    for k in (0.25, 0.5, 0.75, 1.0, 1.5):
        spalte = [tab[(g, k)] for g in (100, 370, 1000)]
        assert all(a < b for a, b in zip(spalte, spalte[1:]))
    assert tab[(370, 0.25)] / 0.25 > tab[(370, 0.5)] / 0.5 > tab[(370, 1.5)] / 1.5   # δ*/k fällt
    assert not math.isclose(tab[(370, 0.5)], 2.45, abs_tol=0.01)                      # 2,45 war die Rasterzahl, exakt ist 2,4375


@pytest.mark.parametrize("delta", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
def test_bestes_k_ist_delta_halbe_gegen_rastersuche(delta):
    k_opt, a_opt = A.best_k(delta, 370)
    ks = np.arange(0.5 * delta - 0.1, 0.5 * delta + 0.1 + 1e-9, 0.005)
    ks = ks[ks > 0.02]
    werte = [A.arl1_tuned(float(k), 370, delta, 100) for k in ks]
    i = int(np.argmin(werte))
    assert abs(float(ks[i]) - 0.5 * delta) <= 0.011          # k*/δ = 0,5 (Raster 0,005)
    assert abs(k_opt / delta - 0.5) < 0.02
    assert a_opt == pytest.approx(min(werte), rel=1e-6)


def test_arl1_optimum_am_gitter_nachgerechnet():
    """ARL1* = 8,5730 bei δ = 1 auch aus dem Gitter (Richardson) bei der abgestimmten Schwelle."""
    h = A.cusum_tune(0.5, 370)
    assert A.grid_arl_richardson(0.5, h, 1.0) == pytest.approx(8.5730, abs=5e-4)


def test_lorden_zeilen_doppelte_knotenzahl_und_gitter():
    for delta, gamma in ((1.0, 370), (2.0, 10), (0.5, 1000), (1.0, 1e6)):
        row = A.lorden_row(delta, gamma)
        n2 = 2 * (200 if gamma >= 1e4 else A.DEFAULT_NODES)
        h2 = A.cusum_tune(delta / 2, gamma, n2)
        assert row["h"] == pytest.approx(h2, rel=1e-6)
        assert row["J"] == pytest.approx(A.cusum_arl(delta / 2, h2, delta, n2), rel=1e-6)
    r = A.lorden_row(1.0, 370)
    assert A.grid_arl_richardson(0.5, r["h"], 1.0) == pytest.approx(r["J"], abs=5e-4)


def test_naeherung_ist_keine_schranke_vorzeichenwechsel_bei_delta_2():
    """δ = 1 und 0,5: bei allen acht γ liegt ln γ / I über J; δ = 2: bei γ = 10 und 30 darunter, ab 100 darüber."""
    gammas = (10, 30, 100, 370, 1000, 1e4, 1e5, 1e6)
    for delta in (0.5, 1.0):
        assert all(A.lorden_row(delta, g)["approx"] > A.lorden_row(delta, g)["J"] for g in gammas)
    zeichen = [A.lorden_row(2.0, g)["approx"] > A.lorden_row(2.0, g)["J"] for g in gammas]
    assert zeichen == [False, False, True, True, True, True, True, True]
    d10, d30 = (A.lorden_row(2.0, g) for g in (10, 30))
    assert d10["J"] - d10["approx"] == pytest.approx(0.157, abs=5e-4) and d30["J"] - d30["approx"] == pytest.approx(0.009, abs=5e-4)


def test_differenz_der_naeherung_aendert_sich_zwischen_1e5_und_1e6_kaum():
    for delta, grenze in ((0.5, 0.0064 * 1.2), (1.0, 0.0064)):
        d5 = (lambda r: r["J"] - r["approx"])(A.lorden_row(delta, 1e5))
        d6 = (lambda r: r["J"] - r["approx"])(A.lorden_row(delta, 1e6))
        assert abs(d5 - d6) <= grenze


def test_shewhart_gegen_cusum_bei_gamma_1e6_und_der_zuschlag_je_zehnerpotenz():
    """δ = 1: bei γ = 10^6 braucht die Shewhart-Karte 11465,574 Beobachtungen gegen 24,2999 beim CUSUM (Verhältnis 471,8); der CUSUM wächst je Zehnerpotenz von γ
    um ln(10) / I = 4,605 (δ = 1, I = 1/2), ab γ = 10^5 auf 0,01 genau; das Verhältnis wächst mit γ."""
    zeilen = {g: A.lorden_row(1.0, g) for g in (1e2, 1e4, 1e5, 1e6)}
    assert zeilen[1e6]["shewhart"] == pytest.approx(11465.574, abs=5e-3)
    assert zeilen[1e6]["J"] == pytest.approx(24.2999, abs=5e-4)
    assert zeilen[1e6]["shewhart"] / zeilen[1e6]["J"] == pytest.approx(471.8, abs=0.05)
    assert zeilen[1e6]["J"] - zeilen[1e5]["J"] == pytest.approx(math.log(10) / 0.5, abs=0.01)
    quot = [zeilen[g]["shewhart"] / zeilen[g]["J"] for g in (1e2, 1e4, 1e6)]
    assert quot[0] < quot[1] < quot[2]
