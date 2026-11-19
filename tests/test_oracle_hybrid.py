"""Orakel: der CUSUM mit zusätzlicher Shewhart-Grenze (Alarm bei S >= h oder x > c) auf anderen Rechenwegen.

* Grenzfall c = unendlich: das Hybrid-Gitter gleicht dem reinen CUSUM der Integralgleichung.
* Simulation (SplitMix64, eigene Rekursion im Test) als Band von 4 Standardfehlern gegen das Gitter (Richardson aus m und 2m).
* Die Restlauflänge L(u) fällt in u: der schlechteste Start ist S = 0, die zero-state-Zahl ist damit das Lorden-Maß (am Gitter gemessen, nicht bewiesen).
Modell: einseitig, k = 0,5, unabhängige Beobachtungen N(delta, 1), Ziel 0, sigma = 1; Zähleinheit Beobachtungen bis zum Alarm."""

import numpy as np
import pytest

import cus_arl as A
import cus_chart as K
from cus_rng import SplitMix64


def test_grenzfall_c_unendlich_gleich_reiner_cusum():
    h = A.cusum_tune(0.5, 370)
    assert A.hybrid_arl_grid(0.5, h, 0.0, 1e9, 640) == pytest.approx(370.0, abs=0.01)
    for delta in (0.5, 1.0, 2.0, 4.0):
        assert A.hybrid_arl(0.5, h, delta, 1e9, 320) == pytest.approx(A.cusum_arl(0.5, h, delta), rel=2e-5)


def test_eine_zusatzgrenze_verkuerzt_die_arl0_bei_gleichem_h():
    assert A.hybrid_arl_grid(0.5, 4.0, 0.0, 3.0, 320) < A.hybrid_arl_grid(0.5, 4.0, 0.0, 1e9, 320)
    assert A.hybrid_arl_grid(0.5, 4.0, 0.0, 3.0, 320) < 1 / float(A.ndtr(-3.0))   # und liegt unter der ARL0 der Grenze allein (740,8)


@pytest.mark.parametrize(("c", "h_soll"), [(4.0, 4.10023), (3.5, 4.14688), (3.0, 4.59761)])
def test_abgestimmte_schwelle_je_grenze_handwerte_der_vormessung(c, h_soll):
    h = A.hybrid_tune(0.5, c, 370)
    assert h == pytest.approx(h_soll, abs=5e-5)
    # h wird am Gitter m = 320 abgestimmt; der Richardson-Wert (m = 320 und 640) liegt um den Gitterfehler (rund 3e-5 relativ) daneben
    assert A.hybrid_arl(0.5, h, 0.0, c) == pytest.approx(370, abs=0.05)


def test_zu_kleine_grenze_ist_nicht_abstimmbar():
    """c = 2,5 hat allein ARL0 = 161 < 370: mit dieser Grenze ist ARL0 = 370 für kein h erreichbar (Meldung statt Zahl)."""
    assert 1 / float(A.ndtr(-2.5)) == pytest.approx(161, abs=0.5)
    with pytest.raises(A.NotTunable):
        A.hybrid_tune(0.5, 2.5, 370)
    assert A.hybrid_tune(0.5, 3.0, 370) > 0


def _simuliere(seed, runs, delta, c, h, k=0.5):
    rng = SplitMix64(seed)
    laengen = []
    for _ in range(runs):
        s = 0.0
        t = 0
        while True:
            t += 1
            x = K.normal_from(rng) + delta
            s = max(0.0, s + x - k)
            if s >= h or x > c:
                break
        laengen.append(t)
    return K.mean_se(laengen)


@pytest.mark.parametrize(("c", "delta", "runs", "seed"), [(3.5, 4.0, 20000, 1), (3.0, 2.0, 20000, 2), (3.5, 0.5, 4000, 3), (3.5, 1.0, 8000, 4)])
def test_hybrid_gitter_gegen_simulation(c, delta, runs, seed):
    h = A.hybrid_tune(0.5, c, 370)
    exakt = A.hybrid_arl(0.5, h, delta, c)
    m, se = _simuliere(seed, runs, delta, c, h)
    assert abs(m - exakt) <= 4 * se, (c, delta, m, se, exakt)


@pytest.mark.parametrize("c", [3.5, 3.0])
def test_restlaufaenge_faellt_in_u_schlechtester_start_ist_null(c):
    h = A.hybrid_tune(0.5, c, 370)
    for delta in (0.0, 1.0):
        r = A.hybrid_matrix(0.5, h, delta, c, 640)
        lv = np.linalg.solve(np.eye(640) - r, np.ones(640))
        assert np.all(np.diff(lv) <= 1e-9)
        assert lv[0] == pytest.approx(A.hybrid_arl_grid(0.5, h, delta, c, 640), rel=1e-12)


def test_handwerte_der_tabelle_g_der_vormessung():
    """c = 3,5: δ = 4 auf 1,3281 (reiner CUSUM 1,7443), δ = 0,5 auf 27,993 (reiner CUSUM 27,675); c = 3,0: δ = 0,5 auf 30,927."""
    pur = A.hybrid_tune(0.5, 1e9, 370)
    h35, h30 = A.hybrid_tune(0.5, 3.5, 370), A.hybrid_tune(0.5, 3.0, 370)
    assert A.hybrid_arl(0.5, pur, 4.0, 1e9) == pytest.approx(1.7443, abs=5e-4)
    assert A.hybrid_arl(0.5, h35, 4.0, 3.5) == pytest.approx(1.3281, abs=5e-4)
    assert A.hybrid_arl(0.5, pur, 0.5, 1e9) == pytest.approx(27.6745, abs=5e-4)
    assert A.hybrid_arl(0.5, h35, 0.5, 3.5) == pytest.approx(27.9930, abs=5e-4)
    assert A.hybrid_arl(0.5, h30, 0.5, 3.0) == pytest.approx(30.9266, abs=5e-4)
