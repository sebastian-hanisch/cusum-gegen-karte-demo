"""Claims: jede Zahl der README steht hier (DEMO-PLAYBOOK §4 und §11a); die Tabelle „Zahl → Test“ der README nennt, welcher Test welche Zahl prüft.

Modell: unabhängige Beobachtungen N(delta, 1), Ziel 0, sigma = 1 bekannt; Zähleinheit Beobachtungen bis zum Alarm; zero-state (= Lorden-Maß für beide Karten); einseitig,
wenn nichts anderes dasteht; ARL0 = 370 ist das gesetzte Ziel (nicht 370,398). Zufallsgrößen (Simulation, SplitMix64) als Band bzw. mit festem Zufalls-Seed.
Tabellen der README: Text gegen Text mit den Ausgaben der App-Funktionen (Marke `<!-- tabelle:NAME -->`); die Werte selbst prüfen die Orakel-Tests gegen andere Rechenwege."""

import math
import re
from pathlib import Path

import pytest

import cus_arl as A
import cus_constants as C
import cus_evaluation as E
import cus_handbeispiel as HB
import cus_reference as REF
import cus_texts as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _block(name: str) -> str:
    m = re.search(rf"<!-- tabelle:{name} -->\n(.*?)\n<!-- /tabelle -->", README, re.S)
    assert m, name
    return m.group(1).strip()


# ----------------------------------------------------------------------------- Kernbotschaft
def test_kernbotschaft():
    ex = E.exact1(1.0, 370, 0.5)
    assert E.de(ex.arl1_cusum, 3) == "8,573" and E.de(ex.arl1_shewhart, 3) == "26,746"
    assert E.de(A.crossover(0.5, 370), 4) == "2,4375"
    assert E.de(A.crossover(0.5, 100), 2) == "2,09" and E.de(A.crossover(0.5, 1000), 2) == "2,69"
    assert E.de(A.crossover(0.25, 370), 2) == "2,03" and E.de(A.crossover(1.5, 370), 2) == "3,80"
    e4 = E.exp2("einseitig", 0.5, 370)
    d4 = next(row for row in e4.table if row[0] == 4.0)
    assert E.de(d4[2], 3) == "1,744" and E.de(d4[1], 3) == "1,126"
    assert E.de(REF.twosided_cell(0.5, 370)["umschlag"], 2) == "2,61"
    assert E.de(100 * ex.fa_shewhart, 2) == "23,71" and E.de(100 * ex.fa_cusum, 2) == "23,04"
    e3 = E.exp3(1.0, 0.5, 370)
    assert E.de(max(v for row in e3.table for v in row if v is not None), 2) == "2,99"        # größtes Verhältnis aller CUSUM-Zeilen: k = 1,5, δ = 0,5
    assert sum(1 for d, k in zip(e3.table_deltas, e3.k_opt_by_delta) if abs(k / d - 0.5) < 0.02) == 8
    assert E.de(A.cusum_tune(0.5, 370), 4) == "4,0954" and E.de(A.cusum_tune(0.5, 370.398), 4) == "4,0965"
    assert E.de(A.shewhart_c(370, two=True), 4) == "2,9997"
    assert E.de(REF.twosided_cell(0.5, 370)["h"], 4) == "4,7738"
    assert A.grid2_arl_richardson(0.5, 4.7749, 0.0) == pytest.approx(370.398, abs=0.02)     # zweiseitig 4,7749 für 370,398
    assert E.de(A.cusum_arl(0.5, 4.77, 0.0), 2) == "737,12"
    # die Zusatzregel „δ = 2 und γ ≤ 30 darunter, δ = 1 bei allen γ darüber“ steht in test_lorden_naeherung


def test_arl_zahlen_der_integralgleichung_aus_der_methodik():
    assert [E.de(A.cusum_arl(0.5, h, d), 7) for h, d in ((4.0, 0.0), (4.0, 1.0), (5.0, 0.0), (5.0, 1.0))] == ["335,3675776", "8,3832021", "930,8870121", "10,3759753"]


# ----------------------------------------------------------------------------- Tabellen der README = Ausgaben der App-Funktionen
def test_readme_tabellen_gleich_app_tabellen():
    assert _block("vergleich") == T.table_exp2(E.exp2("einseitig", 0.5, 370)).strip()
    assert _block("umschlag") == T.table_umschlag(E.umschlag_tabelle()).strip()
    assert _block("zweiseitig") == T.table_exp2(E.exp2("zweiseitig", 0.5, 370)).strip()
    assert _block("k") == T.table_exp3(E.exp3(1.0, 0.5, 370)).strip()
    for name, delta in (("lorden05", 0.5), ("lorden10", 1.0), ("lorden20", 2.0)):
        assert _block(name) == T.table_exp4(E.Exp4(delta, 0, E.exp4(delta, 370).rows)).strip(), name
    assert _block("hybrid") == T.table_hybrid(REF.hybrid_table()).strip()


def test_tabellenzellen_liegen_nicht_auf_einer_rundungsgrenze():
    """Plattformfest: keine Zelle der README-Tabellen liegt so dicht an einer Rundungsgrenze, dass die letzte Binärstelle die gedruckte Ziffer kippen könnte."""
    zellen = []
    for delta in (0.5, 1.0, 2.0):
        for r in E.exp4(delta, 370).rows:
            zellen += [(r["h"], 4), (r["J"], 4), (r["approx"], 4), (r["J"] / r["approx"], 4), (r["J"] - r["approx"], 3), (r["shewhart"], 3)]
    for row in E.exp2("einseitig", 0.5, 370).table:
        zellen += [(v, 3) for v in row[1:] if v is not None]
    e3 = E.exp3(1.0, 0.5, 370)
    zellen += [(v, 3) for row in e3.table for v in row] + [(v, 3) for v in e3.table_shewhart] + [(v, 3) for v in e3.k_opt_by_delta]
    for arl0, werte in E.umschlag_tabelle().items():
        zellen += [(v, 4) for v in werte]
    for x, nd in zellen:
        frac = abs(x * 10 ** nd - math.floor(x * 10 ** nd) - 0.5)
        assert frac > 1e-6, (x, nd)


def test_annahmentabelle_in_app_und_readme_gleich():
    m = re.search(r"<!-- annahmen -->\n(.*?)\n<!-- /annahmen -->", README, re.S)
    assert m and m.group(1).strip() == T.ASSUMPTIONS_MARKDOWN.strip()


# ----------------------------------------------------------------------------- Verteilung und Fehlalarm-Seite
def test_verteilung_und_fehlalarm_seite():
    ex = E.exact1(1.0, 370, 0.5)
    assert (ex.median_cusum, ex.q95_cusum, ex.median_shewhart, ex.q95_shewhart) == (7, 18, 19, 79)
    pmf = A.length_distribution(A.grid_matrix(0.5, ex.h, 1.0, 400))
    assert E.de(A.cdf_at(pmf, 10), 4) == "0,7386"
    pmf0 = A.length_distribution(A.grid_matrix(0.5, ex.h, 0.0, 300), tmax_cap=20000, eps=1e-6)
    assert A.quantile(pmf0, 0.5) == 258 and E.shewhart_quantile(1 / 370, 0.5) == 257           # Median der Fehlalarm-Lauflänge: 258 gegen 257
    assert 100 * ex.fa_shewhart == pytest.approx(23.71, abs=5e-3) and 100 * ex.fa_cusum == pytest.approx(23.04, abs=5e-3)
    assert 300 * ex.fa_cusum == pytest.approx(69.1, abs=0.05) and 300 * ex.fa_shewhart == pytest.approx(71.1, abs=0.05)


def test_simulation_zufalls_seeds_35_und_2026():
    ex = E.exact1(1.0, 370, 0.5)
    erwartet = {35: (8.65, 0.25, 27.25, 1.48, 56, 58, 217, 48, 35), 2026: (9.02, 0.28, 29.30, 1.77, 75, 79, None, None, None)}
    for seed, (mc, sc, ms, ss, fc, fs, first_c, first_s, tied) in erwartet.items():
        sim = E.sim1(seed, 300, 1.0, 370, 0.5, ex.h, ex.c, ex.arl1_shewhart)
        assert sim.runs == 300 and not sim.capped
        assert sim.mean_cusum == pytest.approx(mc, abs=0.005) and sim.se_cusum == pytest.approx(sc, abs=0.005)
        assert sim.mean_shewhart == pytest.approx(ms, abs=0.005) and sim.se_shewhart == pytest.approx(ss, abs=0.005)
        assert (sim.fa_cusum, sim.fa_shewhart) == (fc, fs)
        if first_c is not None:
            assert (sim.first_cusum, sim.first_shewhart, sim.tied) == (first_c, first_s, tied)
        # beide Seeds im Band der exakten Größe (4 Standardfehler)
        assert abs(sim.mean_cusum - ex.arl1_cusum) <= 4 * sim.se_cusum and abs(sim.mean_shewhart - ex.arl1_shewhart) <= 4 * sim.se_shewhart


def test_abdeckung_ueber_40_zufalls_seeds():
    ex = E.exact1(1.0, 370, 0.5)
    n_c = n_s = fa_c = fa_s = 0
    for seed in range(1, 41):
        sim = E.sim1(seed, 300, 1.0, 370, 0.5, ex.h, ex.c, ex.arl1_shewhart)
        n_c += abs(sim.mean_cusum - ex.arl1_cusum) <= 2 * sim.se_cusum
        n_s += abs(sim.mean_shewhart - ex.arl1_shewhart) <= 2 * sim.se_shewhart
        fa_c += abs(sim.fa_cusum / 300 - ex.fa_cusum) <= 2 * math.sqrt(ex.fa_cusum * (1 - ex.fa_cusum) / 300)
        fa_s += abs(sim.fa_shewhart / 300 - ex.fa_shewhart) <= 2 * math.sqrt(ex.fa_shewhart * (1 - ex.fa_shewhart) / 300)
    assert (n_c, n_s, fa_c, fa_s) == (40, 39, 38, 39)
    assert min(n_c, n_s, fa_c, fa_s) >= 34      # Band: bei 95 % Überdeckung sind 38 von 40 zu erwarten, unter 34 wäre es ein Befund


# ----------------------------------------------------------------------------- k
def test_bestes_k_und_der_preis_eines_falschen_k():
    e = E.exp3(1.0, 0.5, 370)
    assert [round(a, 3) for a in e.arl1_opt_by_delta] == [57.377, 23.668, 13.224, 8.573, 4.572, 2.917, 1.571, 1.121]
    assert all(abs(k / d - 0.5) < 0.02 for d, k in zip(e.table_deltas, e.k_opt_by_delta))
    cus_rows = e.table
    assert round(max(v for row in cus_rows for v in row), 3) == 2.991 and round(cus_rows[0][-1], 3) == 2.043   # k = 1,5 bei δ = 0,5; k = 0,25 bei δ = 4
    assert E.de(max(v for v, d in zip(cus_rows[1], e.table_deltas) if d <= 3.0), 3) == "1,427"
    assert E.de(max(v for v, d in zip(e.table_shewhart, e.table_deltas) if d <= 3.0), 3) == "3,756"
    assert round(cus_rows[1][e.table_deltas.index(1.0)], 3) == 1.0 and round(cus_rows[0][e.table_deltas.index(1.0)], 3) == 1.128 and round(cus_rows[4][e.table_deltas.index(1.0)], 3) == 2.062


# ----------------------------------------------------------------------------- Lorden-Näherung
def test_lorden_naeherung():
    gammas = C.LN_GAMMAS
    for delta in (0.5, 1.0):
        rows = E.exp4(delta, 370).rows
        assert all(r["approx"] > r["J"] for r in rows) and len(rows) == 8 and tuple(r["gamma"] for r in rows) == gammas
    rows2 = E.exp4(2.0, 370).rows
    assert [r["approx"] < r["J"] for r in rows2] == [True, True, False, False, False, False, False, False]
    assert E.de(rows2[0]["J"] - rows2[0]["approx"], 3) == "0,157" and E.de(rows2[1]["J"] - rows2[1]["approx"], 3) == "0,009"
    r1 = E.exp4(1.0, 370).rows
    assert E.de(r1[-1]["shewhart"], 3) == "11 465,574" and E.de(r1[-1]["J"], 4) == "24,2999" and E.de(r1[-1]["shewhart"] / r1[-1]["J"], 1) == "471,8"
    r05 = E.exp4(0.5, 370).rows
    diff = {0.5: [r["J"] - r["approx"] for r in r05[-2:]], 1.0: [r["J"] - r["approx"] for r in r1[-2:]]}
    assert [E.de(x, 4) for x in diff[0.5]] == ["−24,6006", "−24,6070"] and [E.de(x, 4) for x in diff[1.0]] == ["−3,3307", "−3,3311"]
    assert E.de(abs(diff[0.5][1] - diff[0.5][0]), 4) == "0,0064" and E.de(abs(diff[1.0][1] - diff[1.0][0]), 4) == "0,0005"


# ----------------------------------------------------------------------------- zweiseitig
def test_zweiseitig_umschlag_und_zaehlung():
    z = REF.twosided_cell(0.5, 370)
    assert E.de(z["umschlag"], 4) == "2,6125" and E.de(A.crossover(0.5, 370), 4) == "2,4375"
    d = REF.twosided_deltas()
    i1 = d.index(1.0)
    assert E.de(z["arl1_cusum"][i1] / z["arl1_shewhart"][i1], 3) == "0,226"
    ein = E.exp2("einseitig", 0.5, 370)
    assert E.de(next(r for r in ein.table if r[0] == 1.0)[4], 3) == "0,321"
    assert E.de(z["arl1_shewhart"][i1], 3) == "43,861" and E.de(z["arl1_cusum"][i1], 3) == "9,925"
    assert E.de(z["c_shewhart"], 4) == "2,9997"
    # Hälftenregel: 369,9898 gegen 369,9873 (h = 4,7738); der Rest ist Gitterfehler der 2D-Kette
    h = 4.7738
    zwei, einseitig = A.grid2_arl_richardson(0.5, h, 0.0), A.cusum_arl(0.5, h, 0.0)
    assert E.de(zwei, 4) == "369,9898" and E.de(einseitig / 2, 4) == "369,9873" and abs(zwei - einseitig / 2) / zwei == pytest.approx(7e-6, abs=2e-6)


def test_gemischte_quellentabelle():
    """Dieselbe Zelle (k = 0,5, h = 4, δ = 0,25) ist zweiseitig 74,2241 und einseitig 77,0785; (h = 5, δ = 0,5): 37,9962 und 38,0096."""
    assert E.de(A.grid2_arl_richardson(0.5, 4.0, 0.25), 4) == "74,2241" and E.de(A.cusum_arl(0.5, 4.0, 0.25), 4) == "77,0785"
    assert E.de(A.grid2_arl_richardson(0.5, 5.0, 0.5), 4) == "37,9962" and E.de(A.cusum_arl(0.5, 5.0, 0.5), 4) == "38,0096"


# ----------------------------------------------------------------------------- Hybrid
def test_hybrid_zahlen_der_readme():
    t = REF.hybrid_table()
    pur = next(z for z in t["zeilen"] if z["c"] is None)["arl1"]
    c35 = next(z for z in t["zeilen"] if z["c"] == 3.5)
    c30 = next(z for z in t["zeilen"] if z["c"] == 3.0)["arl1"]
    i05, i1, i4 = (t["deltas"].index(d) for d in (0.5, 1.0, 4.0))
    assert E.de(100 * (c35["arl1"][i05] / pur[i05] - 1), 1) == "1,2" and E.de(100 * (c35["arl1"][i1] / pur[i1] - 1), 1) == "0,5"
    assert E.de(100 * (c30[i05] / pur[i05] - 1), 1) == "11,8"
    assert E.de(c35["arl1"][i4] / pur[i4], 3) == "0,761" and E.de(c30[i4] / pur[i4], 3) == "0,677"
    assert E.de(c35["arl0_gitter"], 2) == "370,01"
    assert E.de(1 / float(A.ndtr(-2.5)), 0) == "161"


def test_handbeispiel_zahlen_der_readme():
    assert HB.arl_cusum(HB.P0) == HB.arl_shewhart(HB.P0) == 20 and HB.cusum_run_lengths(HB.P0) == [20, HB.F(35, 2), HB.F(25, 2)]
    assert (E.de(float(HB.arl_cusum(HB.P_MITTEL)), 4), HB.arl_shewhart(HB.P_MITTEL)) == ("7,0638", 10)
    assert (E.de(float(HB.arl_cusum(HB.P_GROSS)), 4), E.de(float(HB.arl_shewhart(HB.P_GROSS)), 4)) == ("2,4463", "1,5385")
    p = (HB.F(1, 20), HB.F(1, 10), HB.F(1, 5), HB.F(2, 5), HB.F(1, 4))
    assert (E.de(float(HB.arl_cusum(p)), 4), HB.arl_shewhart(p)) == ("4,0749", 4)
    assert HB.cusum_pfad((3, 3, 3, 0, 4)) == [1, 2, 3, 1, 3] and HB.alarm_times((3, 3, 3, 0, 4)) == (5, 3) and HB.alarm_times((0, 4, 3, 1)) == (2, 3)
    assert 5 + 25 + 125 + 625 + 3125 == 3905


# ----------------------------------------------------------------------------- Presets und Budget
def test_zahlen_der_preset_hilfetexte():
    klein = E.exact1(0.5, 370, 0.5)
    assert klein.arl1_shewhart / klein.arl1_cusum == pytest.approx(3.2, abs=0.05)                  # „das 3,2fache“
    gross = E.exact1(4.0, 370, 0.5)
    assert gross.arl1_cusum / gross.arl1_shewhart == pytest.approx(1.55, abs=0.005)               # „das 1,55fache“
    assert E.de(A.crossover(0.5, 370), 4) == "2,4375"
    e3 = E.exp3(0.5, 1.5, 370)
    assert E.de(e3.ratio_k, 2) == "2,99" and E.de(e3.arl1_k, 2) == "70,80" and E.de(e3.arl1_opt, 2) == "23,67" and E.de(e3.k_opt, 2) == "0,25"
    assert E.de(A.crossover(0.5, 100), 2) == "2,09" and E.de(A.crossover(0.5, 370), 2) == "2,44"
    z = REF.twosided_cell(0.5, 370)
    ein = E.exp2("einseitig", 0.5, 370)
    assert E.de(z["h"], 4) == "4,7738" and E.de(z["c_shewhart"], 4) == "2,9997"
    assert E.de(z["arl1_cusum"][REF.twosided_deltas().index(1.0)] / z["arl1_shewhart"][REF.twosided_deltas().index(1.0)], 3) == "0,226"
    assert E.de(next(r for r in ein.table if r[0] == 1.0)[4], 3) == "0,321"
    r = E.exp4(2.0, 10).rows[0]
    assert E.de(r["approx"], 4) == "1,1513" and E.de(r["J"], 4) == "1,3083"


def test_rechenbudget():
    assert C.SIM_BUDGET == 1_000_000
    e = E.exact1(0.25, 5000, 0.5)
    assert E.de(e.arl1_shewhart, 1) == "1 996,9"
    assert E.max_runs(e.arl1_shewhart, 500) == 470
    assert E.max_runs(E.exact1(1.0, 370, 0.5).arl1_shewhart, 500) == 500
