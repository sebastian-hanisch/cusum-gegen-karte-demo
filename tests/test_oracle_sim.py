"""Orakel: Simulation (SplitMix64, cus_chart) gegen die exakten Größen, jeweils als Band von 4 Standardfehlern (Zufallsergebnisse nie exakt, DEMO-PLAYBOOK §5).

Geprüft wird der Simulationscode der App (cus_chart.simulate_delays und simulate_false_alarms) gegen zwei andere Wege: die Integralgleichung bzw. das Gitter
(CUSUM) und die geschlossene Formel 1/p (Shewhart-Karte). Die Fehlalarm-Lauflänge ARL0 ist hier 100 (kurze Läufe halten die Testzeit klein); die Abstimmung h
auf ARL0 selbst wird in test_oracle_arl.py geprüft. Zufalls-Seeds sind fest; die exakten Anteile (23,04 % und 23,71 %) sind Messwerte der Kette, nicht der Simulation.
Modell: unabhängige Beobachtungen N(delta, 1), Ziel 0, sigma = 1; Zähleinheit Beobachtungen bis zum Alarm; der Sprung liegt vor der ersten Beobachtung (nu = 0)."""

import math

import pytest

import cus_arl as A
import cus_chart as K

ARL0 = 100
RUNS = 1500
ZELLEN = [(d, k) for d in (0.0, 0.5, 1.0, 2.0) for k in (0.25, 0.5)]


@pytest.mark.parametrize(("delta", "k"), ZELLEN)
def test_simulierte_arl_im_band_der_exakten_arl(delta, k):
    h = A.cusum_tune(k, ARL0)
    c = A.shewhart_c(ARL0)
    ts, tc, _ = K.simulate_delays(1000 + int(100 * delta) + int(10 * k), RUNS, delta, c, k, h, 200_000)
    exakt_c = A.cusum_arl(k, h, delta)
    exakt_s = A.shewhart_arl(c, delta)
    for name, werte, exakt in (("CUSUM", tc, exakt_c), ("Shewhart", ts, exakt_s)):
        m, se = K.mean_se(werte)
        assert abs(m - exakt) <= 4 * se, (name, delta, k, m, se, exakt)


def test_fehlalarmanteil_vor_beobachtung_100_im_band_der_exakten_anteile():
    """ARL0 = 370: exakt 23,04 % (CUSUM, k = 0,5) und 23,71 % (Shewhart); beide Karten laufen in der Simulation auf denselben Zahlen."""
    k = 0.5
    h = A.cusum_tune(k, 370)
    c = A.shewhart_c(370)
    runs = 3000
    n_s, n_c, _ = K.simulate_false_alarms(77, runs, 100, c, k, h)
    p_s = 1 - (1 - 1 / 370) ** 100
    pmf0 = A.length_distribution(A.grid_matrix(k, h, 0.0, 300), tmax_cap=101)
    p_c = A.cdf_at(pmf0, 100)
    assert p_s == pytest.approx(0.2371, abs=5e-5) and p_c == pytest.approx(0.2304, abs=5e-5)
    for n, p in ((n_s, p_s), (n_c, p_c)):
        se = math.sqrt(p * (1 - p) / runs)
        assert abs(n / runs - p) <= 4 * se


def test_verzoegerungsverteilung_aus_der_kette_gegen_simulation():
    """δ = 1, ARL0 = 370, k = 0,5: P(T <= t) aus der Kette (Gitter m = 300) gegen die Anteile der simulierten Läufe; Mittel gegen ARL1."""
    k, delta = 0.5, 1.0
    h = A.cusum_tune(k, 370)
    c = A.shewhart_c(370)
    runs = 6000
    ts, tc, _ = K.simulate_delays(4242, runs, delta, c, k, h, 200_000)
    pmf = A.length_distribution(A.grid_matrix(k, h, delta, 300))
    p_sh = float(A.ndtr(delta - c))
    for t in (3, 7, 10, 18, 30):
        p = A.cdf_at(pmf, t)
        n = sum(1 for x in tc if x <= t)
        assert abs(n / runs - p) <= 4 * math.sqrt(p * (1 - p) / runs), t
    for t in (5, 19, 40, 79):
        p = 1 - (1 - p_sh) ** t
        n = sum(1 for x in ts if x <= t)
        assert abs(n / runs - p) <= 4 * math.sqrt(p * (1 - p) / runs), t


def test_verteilung_summiert_zu_eins_und_hat_den_mittelwert_der_arl():
    k = 0.5
    h = A.cusum_tune(k, 370)
    for delta in (0.5, 1.0, 2.0):
        pmf = A.length_distribution(A.grid_matrix(k, h, delta, 300))
        assert pmf[0] == 0.0 and abs(pmf.sum() - 1.0) < 1e-8
        mittel = float(sum(t * p for t, p in enumerate(pmf)))
        assert mittel == pytest.approx(A.cusum_arl(k, h, delta), rel=2e-4)


def test_quantile_und_anteil_der_kette_gegen_handwerte():
    """Handwerte der Plan-Vormessung: CUSUM Median 7, 95-%-Quantil 18, P(T <= 10) = 0,7386; Shewhart Median 19, 95-%-Quantil 79 (δ = 1, ARL0 = 370, k = 0,5)."""
    k, delta = 0.5, 1.0
    h = A.cusum_tune(k, 370)
    pmf = A.length_distribution(A.grid_matrix(k, h, delta, 400))
    assert A.quantile(pmf, 0.5) == 7 and A.quantile(pmf, 0.95) == 18
    assert A.cdf_at(pmf, 10) == pytest.approx(0.7386, abs=5e-5)
    p = float(A.ndtr(delta - A.shewhart_c(370)))
    # kleinstes t mit 1 - (1 - p)^t >= q, durch Probieren gesucht (nicht über den Logarithmus)
    for q, soll in ((0.5, 19), (0.95, 79)):
        t = 1
        while 1 - (1 - p) ** t < q:
            t += 1
        assert t == soll


def test_gleicher_zufalls_seed_gleiche_reihen_und_andere_seeds_andere():
    a = K.simulate_delays(5, 50, 1.0, 2.78, 0.5, 4.09, 100000)
    b = K.simulate_delays(5, 50, 1.0, 2.78, 0.5, 4.09, 100000)
    c = K.simulate_delays(6, 50, 1.0, 2.78, 0.5, 4.09, 100000)
    assert a == b and a != c
    assert K.draw_series(9, 30, 1.0, 10) == K.draw_series(9, 30, 1.0, 10)
    assert K.draw_series(9, 30, 1.0, 10) != K.draw_series(10, 30, 1.0, 10)


def test_die_laeufe_sind_ein_strom_ein_kuerzerer_lauf_ist_ein_praefix_des_laengeren():
    """Gleicher Zufalls-Seed: die ersten 20 Läufe einer Simulation mit 40 Läufen sind die Läufe der Simulation mit 20."""
    ganz = K.simulate_delays(11, 40, 1.0, 2.78, 0.5, 4.09, 100000)
    teil = K.simulate_delays(11, 20, 1.0, 2.78, 0.5, 4.09, 100000)
    assert ganz[0][:20] == teil[0] and ganz[1][:20] == teil[1]
