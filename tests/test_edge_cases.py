"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): jede Zeile prüft einen Randfall der eigenen Routinen.

Modell: unabhängige Beobachtungen N(delta, 1), Ziel 0, sigma = 1; Zähleinheit Beobachtungen bis zum Alarm."""

import math

import pytest

import cus_arl as A
import cus_chart as K
import cus_constants as C
import cus_evaluation as E
import cus_scenario as SC
from cus_rng import SplitMix64


def test_delta_null_fehlalarm_zeit_gleich_arl0():
    ex = E.exact1(0.0, 370, 0.5)
    assert ex.arl1_cusum == pytest.approx(370, rel=1e-8) and ex.arl1_shewhart == pytest.approx(370, rel=1e-12)


def test_delta_negativ_einseitig_sehr_lange_lauflaenge_ohne_absturz():
    ex = E.exact1(-1.0, 370, 0.5)
    assert ex.arl1_cusum > 5 * 370 and ex.arl1_shewhart > 1e4
    assert math.isfinite(ex.arl1_cusum) and ex.median_cusum >= 1


def test_k_ueber_der_shewhart_grenze_ist_nicht_abstimmbar_k_2_78_noch():
    """Bei ARL0 = 370 ist k > 2,7818 nicht erreichbar (selbst h → 0 gibt 1 / Φ(−k) > 370); bei k = 2,78 sind es 367,9, also erreichbar."""
    assert 1 / float(A.ndtr(-2.78)) == pytest.approx(367.93, abs=0.02)
    assert A.cusum_tune(2.78, 370) > 0
    with pytest.raises(A.NotTunable, match="nicht erreichbar"):
        A.cusum_tune(2.79, 370)
    assert A.shewhart_c(370) == pytest.approx(2.7818, abs=5e-5)
    # in der App-Reichweite (k <= 2,5): bei ARL0 = 100 geht es ab k > 2,326 nicht mehr
    with pytest.raises(A.NotTunable):
        A.cusum_tune(2.4, 100)
    with pytest.raises(A.NotTunable):
        E.exact1(1.0, 100, 2.4)


def test_exp3_bei_nicht_abstimmbarem_k_gibt_meldung_statt_zahl():
    e = E.exp3(1.0, 2.5, 100)
    assert e.arl1_k is None and e.ratio_k is None and "nicht erreichbar" in e.note
    kind, text = E.verdict_exp3(e)
    assert kind == "warning" and "nicht erreichbar" in text
    assert "nicht abstimmbar" in E.answer_q3(e)
    assert max(e.ks) <= 2.33   # die Kurve endet vor der Grenze


def test_arl0_gleich_1_schwelle_null_und_nicht_abstimmbar():
    assert A.cusum_arl(0.5, 0.0, 1.0) == 1.0           # S >= 0 gilt schon bei der ersten Beobachtung
    with pytest.raises(A.NotTunable):
        A.cusum_tune(0.5, 1.0)
    with pytest.raises(ValueError):
        A.shewhart_c(1.0)


def test_sehr_grosse_arl0_knotenzahl_200_stabil():
    """ARL0 = 10^6: die Knotenzahl-Kontrolle N = 200 gegen 300 stimmt auf 6 Stellen (V11 C)."""
    h = A.cusum_tune(0.5, 1e6, 200)
    assert A.cusum_arl(0.5, h, 1.0, 200) == pytest.approx(A.cusum_arl(0.5, h, 1.0, 300), rel=1e-6)
    assert A.cusum_arl(0.5, h, 0.0, 200) == pytest.approx(1e6, rel=1e-5)


def test_nu_gleich_null_und_nu_groesser_als_die_reihenlaenge():
    h = A.cusum_tune(0.5, 370)
    c = A.shewhart_c(370)
    s0 = SC.make_series(3, 1.0, 0, c, 0.5, h)
    assert len(s0.xs) == C.SERIES_LENGTH
    s_gross = SC.make_series(3, 1.0, 300, c, 0.5, h)        # nu > Reihenlänge: kein Sprung in der gezeigten Reihe
    assert s_gross.xs == SC.make_series(3, 0.0, 300, c, 0.5, h).xs
    assert SC.delay(s_gross.t_shewhart, 300) is None
    assert SC.delay(None, 10) is None and SC.delay(5, 10) is None and SC.delay(11, 10) == 1 and SC.delay(10, 0) == 10


def test_einzelne_beobachtung_und_reihe_ohne_alarm():
    assert K.draw_series(1, 1, 1.0, 0) and len(K.draw_series(1, 1, 1.0, 0)) == 1
    assert K.cusum_alarm(K.draw_series(1, 1, 0.0, 0), 0.5, 4.0) is None
    assert "kein Alarm in 250 Beobachtungen" in E.series_caption(50, None, None)
    gefunden = False
    for seed in range(40):
        s = SC.make_series(seed, 0.25, 200, A.shewhart_c(370), 0.5, A.cusum_tune(0.5, 370))
        if s.t_cusum is None and s.t_shewhart is None:
            gefunden = True
            assert "kein Alarm in 250 Beobachtungen" in E.series_caption(s.nu, s.t_shewhart, s.t_cusum)
            break
    assert gefunden


def test_fehlalarm_vor_dem_sprung_wird_als_solcher_benannt():
    text = E.series_caption(100, 40, 30)
    assert "Fehlalarm bei Beobachtung 40 (vor dem Sprung)" in text and "Fehlalarm bei Beobachtung 30 (vor dem Sprung)" in text
    text = E.series_caption(10, 12, 25)
    assert "Alarm bei Beobachtung 12, Verzögerung 2" in text and "Alarm bei Beobachtung 25, Verzögerung 15" in text
    assert "keine Regel" in text


def test_eine_einzelne_reihe_r_gleich_1_und_das_budget():
    ex = E.exact1(1.0, 370, 0.5)
    sim = E.sim1(7, 1, 1.0, 370, 0.5, ex.h, ex.c, ex.arl1_shewhart)
    assert sim.runs == 1 and math.isnan(sim.se_cusum)
    assert "–" in E.verdict_exp1(1.0, 370, 0.5, ex, sim, 7)[1]
    assert E.max_runs(26.7, 300) == 300                     # Standard: das Budget reicht
    assert E.max_runs(1997.0, 500) < 500                    # δ = 0,25, ARL0 = 5000: gekappt
    assert E.max_runs(1e9, 500) == 10                       # nie unter 10 Reihen, wenn der Regler mehr verlangt
    assert E.max_runs(1e9, 5) == 5                          # ... und nie mehr als verlangt


def test_zweiseitig_ausserhalb_der_eingefrorenen_tabelle_meldung_statt_zahl():
    for k, arl0 in ((0.3, 370), (0.5, 200), (1.5, 370), (0.5, 5000)):
        e = E.exp2("zweiseitig", k, arl0)
        assert not e.available and e.shewhart == () and "eingefroren" in e.note
        kind, text = E.verdict_exp2(e)
        assert kind == "warning" and "keine Zahl" in text
    assert E.exp2("zweiseitig", 0.5, 370).available


def test_k_halbe_delta_knopf_begrenzt_auf_die_reglergrenzen():
    import cus_presets as P

    import streamlit as st
    st.session_state.clear()
    st.session_state["delta_slider"] = 0.25
    P.set_k_half_delta()
    assert st.session_state["k_slider"] == 0.125
    st.session_state["delta_slider"] = 4.0
    P.set_k_half_delta()
    assert st.session_state["k_slider"] == 2.0
    st.session_state.clear()


def test_zufallsstrom_u_gleich_null_wird_nicht_zur_unendlichkeit():
    class _Null(SplitMix64):
        def random(self):
            return 0.0
    assert math.isfinite(K.normal_from(_Null(1)))


def test_arl0_5000_und_delta_0_25_stehen_in_der_reglerreichweite():
    ex = E.exact1(0.25, 5000, 0.5)
    assert ex.arl1_shewhart > 1000 and ex.q95_cusum > ex.median_cusum > 0
    assert math.isfinite(ex.arl1_cusum)
