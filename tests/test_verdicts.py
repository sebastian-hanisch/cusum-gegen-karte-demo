"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt gesetzte Ergebniszeilen, nicht Seeds: so prüft der Test die Grenzen (Umschlag ja oder nein, Näherung über oder unter J)
und jede genannte Zahl direkt."""

import cus_evaluation as E


# ----------------------------------------------------------------------------- Formatierung
def test_dezimalkomma_minus_und_tausendertrenner():
    assert E.de(8.573036, 3) == "8,573" and E.de(-3.254, 3) == "−3,254" and E.de(11465.574, 3) == "11 465,574" and E.de(float("nan")) == "–"
    assert E.dg(1.0) == "1" and E.dg(0.5) == "0,5" and E.dg(0.125) == "0,125" and E.dg(0.0) == "0" and E.dg(2.0) == "2"
    assert E.fach(0.32064) == "0,321fache"


# ----------------------------------------------------------------------------- Experiment 1
def _ex1(arl1_c=8.573, arl1_s=26.746):
    return E.Exact1(delta=1.0, arl0=370, k=0.5, h=4.0954, c=2.7818, arl1_cusum=arl1_c, arl1_shewhart=arl1_s, median_cusum=7, median_shewhart=19, q95_cusum=18, q95_shewhart=79,
                    fa_cusum=0.2304, fa_shewhart=0.2371, pmf_cusum=(0.0, 0.5, 0.5), p_shewhart=0.0374)


def _sim1():
    return E.Sim1(runs=300, capped=False, mean_cusum=8.65, se_cusum=0.25, mean_shewhart=27.25, se_shewhart=1.48, first_cusum=250, first_shewhart=30, tied=20, fa_cusum=56, fa_shewhart=58,
                  delays_cusum=(), delays_shewhart=(), normals=1000)


def test_verdikt_1_nennt_exakte_zahlen_reihen_und_fehlalarm_seite():
    kind, text = E.verdict_exp1(1.0, 370, 0.5, _ex1(), _sim1(), 35)
    assert kind == "success"
    for teil in ("δ = 1, ARL₀ = 370, einseitig, k = 0,5", "**8,573** Beobachtungen bis zum Alarm", "(Median 7, 95-%-Quantil 18)", "**26,746** (Median 19, 95-%-Quantil 79)",
                 "In diesen 300 Reihen (Zufalls-Seed 35)", "8,65 ± 0,25 gegen 27,25 ± 1,48 (Mittel ± Standardfehler)", "bis Beobachtung 100 exakt 23,04 % (CUSUM) gegen 23,71 % (Shewhart)",
                 "in den 300 Reihen ohne Sprung 56 gegen 58", "Lorden-Maß ist für beide Karten der Wert bei ν = 0"):
        assert teil in text, teil


def test_verdikt_1_art_schwelle_cusum_nicht_schneller_ist_eine_warnung():
    assert E.verdict_exp1(4.0, 370, 0.5, _ex1(1.744, 1.126), _sim1(), 35)[0] == "warning"
    assert E.verdict_exp1(1.0, 370, 0.5, _ex1(10.0, 10.0), _sim1(), 35)[0] == "warning"      # gleich: kein Vorsprung
    assert E.verdict_exp1(1.0, 370, 0.5, _ex1(9.99, 10.0), _sim1(), 35)[0] == "success"


# ----------------------------------------------------------------------------- Experiment 2
def _row(d, s, ck, eh):
    return (d, s, ck, eh, ck / s, None if eh is None else eh / s)


def _exp2(zaehlung="einseitig", crossover=2.4375, by=None, rows=None, available=True, h=4.0954):
    rows = rows or (_row(0.75, 47.426, 13.640, 13.224), _row(1.0, 26.746, 8.573, 8.573), _row(3.0, 1.705, 2.231, 1.571), _row(4.0, 1.1256, 1.7443, 1.1214))
    return E.Exp2(zaehlung, 0.5, 370, h, 2.7818, (), (), (), (), rows, crossover, by if by is not None else {100: 2.0914, 370: 2.4375, 1000: 2.6857}, available, "Meldung")


def test_verdikt_2_einseitig_nennt_verhaeltnisse_umschlag_und_gewinnzaehlung():
    kind, text = E.verdict_exp2(_exp2())
    assert kind == "info"
    for teil in ("Einseitig, ARL₀ = 370, k = 0,5, h = 4,0954", "bei δ = 0,75 das 0,288fache, bei δ = 1 das 0,321fache und bei δ = 4 das 1,550fache der Shewhart-Verzögerung",
                 "Ab δ = 2,4375 hat die Shewhart-Karte die kleinere Verzögerung", "2,09 bei 100, 2,69 bei 1000", "Mit k = δ/2 gewinnt der CUSUM in 4 von 4 geprüften δ",
                 "Bei δ = 4 knapp: 1,121 gegen 1,126"):
        assert teil in text, teil


def test_verdikt_2_ohne_umschlag_und_ohne_halbes_k_und_knapp_nur_ab_95_prozent():
    _, text = E.verdict_exp2(_exp2(crossover=None))
    assert "gibt es in diesem Raster nicht" in text and "Umschlag wandert" not in text
    rows = (_row(1.0, 10.0, 5.0, 5.0), _row(4.0, 2.0, 3.0, 1.9))      # 1,9 / 2,0 = 0,95: knapp; 5 / 10: nicht knapp
    _, text = E.verdict_exp2(_exp2(rows=rows))
    assert "gewinnt der CUSUM in 2 von 2" in text and "Bei δ = 4 knapp: 1,900 gegen 2,000" in text
    rows = (_row(1.0, 10.0, 5.0, 5.0), _row(4.0, 2.0, 3.0, 1.8))      # 0,9: nicht knapp
    assert "knapp" not in E.verdict_exp2(_exp2(rows=rows))[1]
    rows = (_row(1.0, 10.0, 5.0, None), _row(4.0, 2.0, 3.0, 2.5))     # k = δ/2 verliert bei δ = 4: 1 von 1 gerechneten ...
    assert "gewinnt der CUSUM in 0 von 1" in E.verdict_exp2(_exp2(rows=rows))[1]


def test_verdikt_2_zweiseitig_nennt_die_zaehlung_und_warnt_vor_dem_mischen():
    rows = (_row(0.5, 155.0752, 35.2534, None), _row(1.0, 43.8596, 9.9246, None), _row(4.0, 1.1885, 1.9568, None))
    kind, text = E.verdict_exp2(_exp2("zweiseitig", 2.6125, {}, rows, h=4.7738))
    assert kind == "info" and text.startswith("Zweiseitig, ARL₀ = 370, k = 0,5, h = 4,7738")
    assert "bei δ = 0,5 das 0,227fache, bei δ = 1 das 0,226fache und bei δ = 4 das 1,646fache" in text
    assert "Ab δ = 2,6125" in text and "Zahlen aus einseitiger und zweiseitiger Zählung dürfen nicht gemischt werden" in text
    assert "Mit k = δ/2" not in text


def test_verdikt_2_nicht_verfuegbar_ist_eine_warnung_mit_der_meldung():
    assert E.verdict_exp2(_exp2(available=False)) == ("warning", "Meldung")


# ----------------------------------------------------------------------------- Experiment 3
def _exp3(arl1_k=70.80, ratio=2.991, **kw):
    deltas = (0.25, 0.5, 3.0, 4.0)
    row05 = (1.427, 1.169, 1.420, 1.556)
    return E.Exp3(0.5, 1.5, 370, (0.25, 1.5), (23.67, 70.80), 0.25, 23.67, arl1_k, ratio if arl1_k else None, 88.89, 3.756, (0.5, 1.5), deltas, (row05, (2.749, 2.991, 1.0, 1.009)), (3.072, 3.756, 1.086, 1.004),
                  (0.125, 0.25, 1.5, 2.0), (57.0, 23.67, 1.57, 1.12), kw.get("note", ""))


def test_verdikt_3_preis_eines_falschen_k_und_die_worst_case_zahlen():
    kind, text = E.verdict_exp3(_exp3())
    assert kind == "info"
    for teil in ("Wahres δ = 0,5, k = 1,5 (für δ = 3 gedacht), ARL₀ = 370, einseitig", "ARL₁ = **70,80** Beobachtungen gegen das Optimum 23,67 bei k = 0,25, das **2,991fache**",
                 "k = 0,5 braucht bei den geprüften δ bis 3 höchstens das 1,427fache", "die Shewhart-Karte höchstens das 3,756fache",
                 "Das beste k ist δ/2 (in 4 von 4 geprüften δ bei dieser ARL₀)", "für bekanntes δ"):
        assert teil in text, teil


def test_verdikt_3_delta_4_zaehlt_nicht_zum_worst_case_und_nicht_abstimmbar_ist_eine_warnung():
    assert E.worst_ratio((1.0, 2.0, 5.0), (1.0, 3.0, 4.0)) == 2.0          # δ = 4 > 3 zählt nicht; Grenze inklusive 3
    assert E.worst_ratio((None, None), (1.0, 2.0)) is None
    kind, text = E.verdict_exp3(_exp3(arl1_k=None, note="k nicht erreichbar"))
    assert (kind, text) == ("warning", "k nicht erreichbar")
    assert E.verdict_exp3(_exp3(arl1_k=None))[1].startswith("Für dieses k")


# ----------------------------------------------------------------------------- Experiment 4
def _rows(approx_minus_j):
    gam = (10, 30, 100, 370, 1000, 10000, 100000, 1000000)
    return tuple({"gamma": g, "h": 1.0, "J": 10.0, "approx": 10.0 + d, "shewhart": 100.0 * g} for g, d in zip(gam, approx_minus_j))


def test_verdikt_4_naeherung_ueber_allen_gamma_keine_schranke_nur_der_grenzwert_wird_nicht_behauptet():
    e = E.Exp4(1.0, 370, _rows((2.11, 2.76, 3.10, 3.25, 3.30, 3.33, 3.33, 3.331)))
    kind, text = E.verdict_exp4(e)
    assert kind == "info"
    for teil in ("δ = 1, k = δ/2 = 0,5, einseitig, zero-state", "bei 8 von 8 geprüften γ über der exakten Verzögerung", "und bei 0 darunter", "Bei γ = 370: J = 10,0000, ln γ / I = 13,2500, Differenz −3,250",
                 "von −2,11 bei γ = 10 auf −3,331 bei γ = 1 000 000", "bei δ = 2 liegt sie für γ = 10 und 30 darunter", "braucht bei γ = 1 000 000 100 000 000,000 Beobachtungen gegen 10,0000 beim CUSUM"):
        assert teil in text, teil


def test_verdikt_4_vorzeichenwechsel_nennt_die_gamma_darunter():
    e = E.Exp4(2.0, 10, _rows((-0.157, -0.009, 0.035, 0.040, 0.041, 0.042, 0.043, 0.044)))
    _, text = E.verdict_exp4(e)
    assert "bei 6 von 8 geprüften γ über" in text and "bei 2 darunter" in text
    assert "Darunter liegt sie bei γ = 10, 30: Die Näherung ist weder obere noch untere Schranke" in text
    assert "Bei γ = 10: J = 10,0000, ln γ / I = 9,8430, Differenz 0,157" in text


# ----------------------------------------------------------------------------- Hybrid
def test_verdikt_hybrid_nennt_gewinn_bei_grossem_und_kosten_bei_kleinem_sprung():
    t = {"deltas": [0.5, 1.0, 2.0, 3.0, 4.0], "zeilen": [
        {"c": None, "h": 4.0, "arl1": [27.6746, 8.5731, 3.4061, 2.2307, 1.7443]},
        {"c": 3.5, "h": 4.1, "arl1": [27.9930, 8.6200, 3.3345, 1.9692, 1.3281]},
        {"c": 3.0, "h": 4.6, "arl1": [30.9266, 9.1562, 3.3038, 1.7765, 1.1805]}]}
    kind, text = E.verdict_hybrid(t)
    assert kind == "info"
    for teil in ("c = 3,5: bei δ = 4 auf das 0,761fache der reinen CUSUM-Verzögerung (1,328 gegen 1,744)", "bei δ = 0,5 um 1,2 % schlechter (27,993 gegen 27,675)",
                 "mit c = 3,0 kostet es 11,8 % bei δ = 0,5 und bringt das 0,677fache bei δ = 4", "ARL₀ = 370 einseitig und k = 0,5"):
        assert teil in text, teil


# ----------------------------------------------------------------------------- Kernfragen und Hilfen
def test_kernfragen_antworten_nennen_zahlen_und_zaehlung():
    assert "das 0,321fache" in E.answer_q1(_ex1(8.573, 26.746)) and "ARL₀ = 370 (einseitig, k = 0,5)" in E.answer_q1(_ex1())
    assert "ab δ = 2,4375 hat die Shewhart-Karte die kleinere Verzögerung" in E.answer_q2(_exp2())
    assert "kippt es nicht" in E.answer_q2(_exp2(crossover=None)) and "nicht abstimmbar" in E.answer_q2(_exp2(available=False))
    q3 = E.answer_q3(_exp3())
    assert "k = 1,5 (für δ = 3 gedacht) bei wahrem δ = 0,5: das 2,991fache" in q3 and "k* = 0,250" in q3
    e = E.Exp4(2.0, 10, _rows((-0.157, -0.009, 0.035, 0.040, 0.041, 0.042, 0.043, 0.044)))
    assert "liegt die Näherung ln γ / I = 9,843 unter der exakten Verzögerung J = 10,000" in E.answer_q4(e)


def test_geometrische_quantile_gegen_probieren():
    for p, q in ((0.0374, 0.5), (0.0374, 0.95), (0.5, 0.5), (0.5, 0.99), (1e-4, 0.5), (0.9, 0.95)):
        t = 1
        while 1 - (1 - p) ** t < q:
            t += 1
        assert E.shewhart_quantile(p, q) == t, (p, q)


def test_klassen_der_histogramme_summieren_zu_eins_und_zaehlen_ab_1():
    assert E.bin_width(40) == 1 and E.bin_width(41) == 2 and E.bin_width(400) == 10
    assert E.histogram([1, 2, 3, 4, 10], width=2, nbins=3) == [0.4, 0.4, 0.0]          # Klassen {1, 2}, {3, 4}, {5, 6}; die 10 fällt heraus
    cus, shew = E.exact_bins((0.0, 0.25, 0.25, 0.5), 0.5, 2, 2)
    assert cus == [0.5, 0.5] and shew == [0.75, 0.1875]                              # 1 - 0,5^2 und 0,5^2 - 0,5^4
