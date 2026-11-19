"""Eigenschaften (je 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed, ORACLE-PLAYBOOK §5): Aussagen, die für jede Reihe oder jede Einstellung gelten müssen.

Modell: unabhängige Beobachtungen N(delta, 1), Ziel 0, sigma = 1; Zähleinheit Beobachtungen bis zum Alarm."""

import math

import numpy as np
import pytest
from scipy.stats import kstest

import cus_arl as A
import cus_chart as K
from cus_rng import SplitMix64


def test_rekursion_ist_das_maximum_der_teilsummen_in_50_zufallsreihen():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(1, 40))
        x = rng.standard_normal(n) + rng.uniform(-1, 1.5)
        k = float(rng.uniform(0, 1.2))
        s = K.cusum_path(list(x), k)[-1]
        z = x - k
        bester = max([0.0] + [float(z[nu:].sum()) for nu in range(n)])
        assert s == pytest.approx(bester, abs=1e-9)


def test_warteschlange_mit_beliebigen_zuwaechsen_gleich_cusum_pfad():
    """Ereignisrechnung (Ankunfts- und Fertigstellungszeiten, reelle Zahlen) gegen die Zuwachsrekursion; der Zuwachs ist Bedienzeit minus Zwischenankunftszeit."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        n = int(rng.integers(1, 60))
        bedien = rng.exponential(0.8, n)
        zwischen = rng.exponential(1.0, n)
        ankunft = np.concatenate([[0.0], np.cumsum(zwischen)])
        fertig = 0.0
        waits = []
        for j in range(n + 1):
            beginn = max(ankunft[j], fertig)
            waits.append(beginn - ankunft[j])
            if j < n:
                fertig = beginn + bedien[j]
        pfad = K.cusum_path(list(bedien - zwischen), 0.0)
        assert waits[1:] == pytest.approx(pfad, abs=1e-9)


def test_verschiebung_und_skalierung_geben_dieselbe_alarmzeit():
    """Rohdaten y = a x + b, mit bekanntem Ziel b und bekanntem σ = a standardisiert, haben dieselben Alarmzeiten wie x."""
    rng = np.random.default_rng(3)
    for _ in range(40):
        x = list(rng.standard_normal(80) + 0.8)
        a, b = float(rng.choice([0.5, 2.0, 4.0])), float(rng.choice([-3.0, 5.0, 12.0]))
        y = [a * v + b for v in x]
        z = [(v - b) / a for v in y]
        assert K.cusum_alarm(z, 0.5, 3.0) == K.cusum_alarm(x, 0.5, 3.0)
        assert K.shewhart_alarm(z, 2.0) == K.shewhart_alarm(x, 2.0)


def test_zweiseitig_ist_in_delta_nach_minus_delta_symmetrisch():
    rng = np.random.default_rng(4)
    for _ in range(30):
        x = list(rng.standard_normal(60) + 0.7)
        up, dn = K.cusum_path_two(x, 0.5)
        up2, dn2 = K.cusum_path_two([-v for v in x], 0.5)
        assert up == dn2 and dn == up2


def test_grenzfall_h_gegen_null_ist_die_shewhart_karte_mit_grenze_k():
    for k, delta in ((0.5, 1.0), (1.0, 2.0), (0.25, 0.5)):
        assert A.cusum_arl(k, 1e-4, delta) == pytest.approx(1 / float(A.ndtr(delta - k)), rel=1e-3)
        assert A.cusum_arl(k, 1e-4, 0.0) == pytest.approx(A.max_arl0(k), rel=1e-3)


def test_arl0_der_zusatzregel_ordnung_h_wird_in_der_abstimmung_monoton():
    hs = [A.cusum_tune(0.5, g) for g in (100, 200, 370, 500, 1000, 2000, 5000)]
    assert all(a < b for a, b in zip(hs, hs[1:]))


def test_restlaufaenge_faellt_in_u_und_der_start_null_ist_der_schlechteste():
    """Das Lorden-Maß des CUSUM ist der Wert ab S = 0: die Verzögerung nach ν Beobachtungen ohne Alarm ist nie größer (Gitter m = 400, δ = 1)."""
    k, delta = 0.5, 1.0
    h = A.cusum_tune(k, 370)
    m = 400
    l1 = np.linalg.solve(np.eye(m) - A.grid_matrix(k, h, delta, m), np.ones(m))
    assert np.all(np.diff(l1) <= 1e-9)
    r0 = A.grid_matrix(k, h, 0.0, m)
    p = np.zeros(m)
    p[0] = 1.0
    verzoegerung = []
    for nu in range(0, 201):
        verzoegerung.append(float(p / p.sum() @ l1))
        p = p @ r0
    assert all(d <= verzoegerung[0] + 1e-12 for d in verzoegerung)
    assert verzoegerung[-1] == pytest.approx(A.quasi_stationary_delay(k, h, delta, m), abs=1e-4)
    assert verzoegerung[0] == pytest.approx(8.5730, abs=5e-4)


def test_quasistationaere_verzoegerung_ist_kleiner_als_zero_state_und_nicht_das_shiryaev_mass():
    """δ = 1, k = 0,5, ARL0 = 370: quasistationär 7,9038 gegen zero-state 8,5730 (V12; V1 mit gerundetem h: 7,9039 und 8,5731)."""
    h = A.cusum_tune(0.5, 370)
    assert A.quasi_stationary_delay(0.5, h, 1.0) == pytest.approx(7.9038, abs=2e-4)
    assert A.cusum_arl(0.5, h, 1.0) == pytest.approx(8.5730, abs=5e-4)


def test_zufallszahlen_der_app_sind_standardnormal():
    rng = SplitMix64(99)
    z = [K.normal_from(rng) for _ in range(20000)]
    assert abs(float(np.mean(z))) < 4 / math.sqrt(20000)
    assert abs(float(np.std(z, ddof=1)) - 1) < 4 / math.sqrt(2 * 20000)
    assert kstest(z, "norm").pvalue > 1e-3


def test_einzelreihe_hat_vor_dem_sprung_mittel_null_und_danach_delta():
    x = K.draw_series(5, 20000, 1.0, 10000)
    assert abs(float(np.mean(x[:10000]))) < 4 / 100
    assert abs(float(np.mean(x[10000:])) - 1.0) < 4 / 100
    assert K.draw_series(5, 20, 1.0, 0) == [v + 1.0 for v in K.draw_series(5, 20, 0.0, 0)]


def test_alarm_zeit_ist_der_erste_index_und_zaehlt_die_alarmbeobachtung_mit():
    assert K.first_index([0, 0, 5, 9], lambda v: v >= 5) == 3
    assert K.first_index([0, 0], lambda v: v >= 5) is None
    assert K.cusum_alarm([3.0, 3.0, 3.0], 2.0, 3.0) == 3          # S = 1, 2, 3
    assert K.shewhart_alarm([0.0, 4.5], 4.0) == 2 and K.shewhart_alarm([4.0], 4.0) is None   # Alarm nur bei x > c, nicht bei x = c
