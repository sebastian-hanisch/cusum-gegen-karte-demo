"""Handwerte in Brüchen (ORACLE-PLAYBOOK §5): das kleine Zählmodell der App (cus_handbeispiel.py), von Hand nachgerechnet.

Jede Erwartung steht hier als Rechnung oder als Bruch, nicht als Ausgabe des Codes. Die Gegenwege sind andere Rechenarten als die des Moduls:
Aufzählung aller Folgen (statt Matrixweg), Summe der Überlebenswahrscheinlichkeiten (statt lineares System), Ereignisrechnung einer Warteschlange (statt Zuwachsrekursion).

Modell: Beobachtung x in {0, ..., 4}; ohne Änderung P0 = (4, 5, 5, 5, 1) / 20; CUSUM k = 2, h = 3 (Zustände 0, 1, 2); Shewhart-Alarm bei x = 4."""

import itertools
from fractions import Fraction as F

import numpy as np

import cus_handbeispiel as HB


def test_uebergangsmatrix_und_alarmwahrscheinlichkeit_von_hand():
    # Zuwachs y = x - 2 in {-2, -1, 0, +1, +2} mit 1/5, 1/4, 1/4, 1/4, 1/20
    von_hand = [[F(7, 10), F(1, 4), F(1, 20)], [F(9, 20), F(1, 4), F(1, 4)], [F(1, 5), F(1, 4), F(1, 4)]]
    assert HB.transition_matrix(HB.P0) == von_hand
    assert [1 - sum(zeile) for zeile in von_hand] == [0, F(1, 20), F(3, 10)]


def test_arl0_beider_karten_ist_20_von_hand():
    l = HB.cusum_run_lengths(HB.P0)
    assert l == [20, F(35, 2), F(25, 2)]
    # Einsetzen in die drei Gleichungen
    assert l[0] == 1 + F(7, 10) * l[0] + F(1, 4) * l[1] + F(1, 20) * l[2]
    assert l[1] == 1 + F(9, 20) * l[0] + F(1, 4) * l[1] + F(1, 4) * l[2]
    assert l[2] == 1 + F(1, 5) * l[0] + F(1, 4) * l[1] + F(1, 4) * l[2]
    assert HB.arl_shewhart(HB.P0) == 20 == 1 / F(1, 20)


def _cusum_alarm_time_by_hand(folge):
    s = 0
    for t, x in enumerate(folge, start=1):
        s = max(0, s + x - HB.K)
        if s >= HB.H:
            return t
    return None


def _enumerate_pmf(tmax):
    """Aufzählung aller 5^t Folgen: P(Alarm genau bei t) als Bruch (anderer Rechenweg als der Matrixweg des Moduls)."""
    out = [F(0)] * tmax
    for t in range(1, tmax + 1):
        for folge in itertools.product(range(5), repeat=t):
            if _cusum_alarm_time_by_hand(folge) == t:
                pr = F(1)
                for x in folge:
                    pr *= HB.P0[x]
                out[t - 1] += pr
    return out


def test_lauflaengenverteilung_matrixweg_gleich_aufzaehlung_aller_folgen():
    assert HB.cusum_pmf(HB.P0, 5) == _enumerate_pmf(5)  # 5 + 25 + 125 + 625 + 3125 Folgen


def test_lauflaengenverteilung_von_hand():
    assert HB.cusum_pmf(HB.P0, 5) == [0, F(11, 400), F(91, 2000), F(1547, 32000), F(18927, 400000)]
    assert HB.shewhart_pmf(HB.P0, 3) == [F(1, 20), F(19, 400), F(361, 8000)]
    # P(Lauflänge = 2) = P(x1 = 3) P(x2 = 4) + P(x1 = 4 ... ) von Hand: 1/4 * 1/20 + 1/20 * 3/10 (Zustand 1, dann y = +2; Zustand 2, dann y = +1 oder +2)
    assert F(1, 4) * F(1, 20) + F(1, 20) * F(3, 10) == F(11, 400)
    # in den ersten zwei Beobachtungen meldet die Shewhart-Karte 39/400, der CUSUM 11/400
    assert sum(HB.shewhart_pmf(HB.P0, 2)) == 1 - F(19, 20) ** 2 == F(39, 400)
    assert sum(HB.cusum_pmf(HB.P0, 2)) == F(11, 400)


def test_beide_karten_ueber_zehn_beobachtungen_der_cusum_meldet_seltener_falsch():
    assert sum(HB.cusum_pmf(HB.P0, 10)) < sum(HB.shewhart_pmf(HB.P0, 10))
    assert abs(float(sum(HB.cusum_pmf(HB.P0, 10))) - 0.3720) < 5e-4 and abs(float(sum(HB.shewhart_pmf(HB.P0, 10))) - 0.4013) < 5e-4


def test_summe_der_ueberlebenswahrscheinlichkeiten_gibt_die_arl_in_gleitkomma():
    """Zweiter Weg für die ARL: ARL = Summe über t >= 0 von P(Lauflänge > t) (Vektoriteration in Gleitkomma, 1500 Schritte)."""
    for p in (HB.P0, HB.P_MITTEL, HB.P_GROSS):
        r = np.array([[float(v) for v in zeile] for zeile in HB.transition_matrix(p)])
        v = np.array([1.0, 0.0, 0.0])
        gesamt = 0.0
        for _ in range(1500):
            gesamt += v.sum()
            v = v @ r
        assert abs(gesamt - float(HB.arl_cusum(p))) < 1e-9


def test_arl1_bei_gleicher_arl0_die_rangfolge_kehrt_sich_um():
    assert HB.arl_cusum(HB.P_MITTEL) == F(2105, 298) and HB.arl_shewhart(HB.P_MITTEL) == 10
    assert HB.arl_cusum(HB.P_GROSS) == F(2620, 1071) and HB.arl_shewhart(HB.P_GROSS) == F(20, 13)
    assert HB.arl_cusum(HB.P_MITTEL) < HB.arl_shewhart(HB.P_MITTEL)       # kleiner bis mittlerer Sprung: CUSUM schneller
    assert HB.arl_cusum(HB.P_GROSS) > HB.arl_shewhart(HB.P_GROSS)         # großer Sprung: Shewhart-Karte schneller
    assert abs(float(F(2105, 298) / 10) - 0.706) < 5e-4 and abs(float(F(2620, 1071) / F(20, 13)) - 1.590) < 5e-4


def test_gegenprobe_nicht_jeder_sprung_der_p4_erhoeht_gibt_dem_cusum_recht():
    p = (F(1, 20), F(1, 10), F(1, 5), F(2, 5), F(1, 4))   # P(x = 4) = 1/4, Mittel 2,7
    assert sum(p) == 1
    assert HB.arl_cusum(p) == F(1740, 427) and HB.arl_shewhart(p) == 4
    assert HB.arl_cusum(p) > HB.arl_shewhart(p)


def test_dominanter_eigenwert_der_alarmfreien_kette():
    r = np.array([[float(v) for v in zeile] for zeile in HB.transition_matrix(HB.P0)])
    lam = max(np.linalg.eigvals(r).real)
    assert abs(lam - 0.945366) < 5e-6 and abs(lam - (1 - 1 / 20)) < 0.01


def test_pfade_von_hand():
    assert HB.cusum_pfad((3, 3, 3, 0, 4)) == [1, 2, 3, 1, 3]      # läuft über den Alarm hinaus weiter
    assert HB.alarm_times((3, 3, 3, 0, 4)) == (5, 3)               # (Shewhart, CUSUM)
    assert HB.cusum_pfad((0, 4, 3, 1)) == [0, 2, 3, 2]
    assert HB.alarm_times((0, 4, 3, 1)) == (2, 3)                  # Gegenstück: Shewhart zuerst
    assert HB.alarm_times((0, 0, 0)) == (None, None)


def test_rekursion_gleich_maximum_der_teilsummen_in_allen_folgen_der_laenge_bis_5():
    for t in range(1, 6):
        for folge in itertools.product(range(5), repeat=t):
            s = HB.cusum_pfad(folge)[-1]
            bester = max([0] + [sum(x - HB.K for x in folge[nu:]) for nu in range(t)])
            assert s == bester, folge


def test_ereignisrechnung_einer_warteschlange_gleich_cusum_pfad_in_allen_3125_folgen_der_laenge_5():
    """Ein-Schalter-System: Kunde j kommt zur Zeit 2 j (Zwischenankunftszeit k = 2) und braucht Bedienzeit x_j; Beginn = max(Ankunft, Fertig des Vorgängers).

    Die Wartezeit des Kunden i + 1 ist der CUSUM-Wert S_i. Das ist eine andere Rechnung (Zeiten statt Zuwächse); „Lindley = CUSUM“ allein wäre nur eine Umbenennung."""
    for folge in itertools.product(range(5), repeat=5):
        fertig, waits = 0, []
        for j in range(len(folge) + 1):
            ankunft = 2 * j
            beginn = max(ankunft, fertig)
            waits.append(beginn - ankunft)
            if j < len(folge):
                fertig = beginn + folge[j]
        assert waits[1:] == HB.cusum_pfad(folge), folge
