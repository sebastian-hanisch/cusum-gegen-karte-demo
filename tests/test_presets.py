"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, und je Preset die erwartete Verdikt-Zeile (kein Preset ohne Wirkung meldet eine Wirkung).

Die Zahlen der Erwartungen sind die Handwerte der Vormessung (Plan, V1 und V11) bzw. Rechnungen dieses Demo-Ordners, die in test_claims.py und den Orakel-Tests gegen andere Wege laufen."""

import pytest

import cus_constants as C
import cus_evaluation as E
import cus_presets as P

STANDARD = next(iter(C.PRESETS))


def _settings(name):
    return E.Settings(**C.PRESETS[name])


def test_jedes_preset_hat_hilfetext_und_alle_schluessel():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_werte_liegen_in_den_reglergrenzen_oder_optionen():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            assert P.clamp_value(spec, preset[key]) == preset[key], (name, key)
        assert preset["k"] / C.K_STEP == pytest.approx(round(preset["k"] / C.K_STEP))      # k liegt auf dem Raster des Reglers
        assert preset["delta"] / C.DELTA_STEP == pytest.approx(round(preset["delta"] / C.DELTA_STEP))


def test_standard_preset_gleich_standardeinstellungen_und_erstes_preset():
    assert _settings(STANDARD) == E.Settings()
    assert STANDARD.startswith("Standard")


def test_setting_specs_decken_jeden_preset_schluessel_und_die_presets_sind_verschieden():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    werte = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)


def test_permalink_begrenzt_und_prueft_optionen():
    spec = P.SETTING_SPECS["delta_slider"]
    assert P.clamp_value(spec, 99.0) == C.DELTA_MAX and P.clamp_value(spec, -1.0) == C.DELTA_MIN
    assert P.clamp_value(P.SETTING_SPECS["arl0_select"], 370) == 370 and P.clamp_value(P.SETTING_SPECS["arl0_select"], 371) is None
    assert P.clamp_value(P.SETTING_SPECS["zaehlung_radio"], "dreiseitig") is None and P.clamp_value(P.SETTING_SPECS["ln_gamma_select"], 10) == 10


# ----------------------------------------------------------------------------- je Preset die erwartete Verdikt-Zeile
def _lines(name):
    s = _settings(name)
    out = {}
    ex = E.exact1(s.delta, s.arl0, s.k)
    out["exp1"] = E.verdict_exp1(s.delta, s.arl0, s.k, ex, E.sim1(s.seed, 50, s.delta, s.arl0, s.k, ex.h, ex.c, ex.arl1_shewhart), s.seed)
    out["exp2"] = E.verdict_exp2(E.exp2(s.zaehlung, s.k, s.arl0))
    out["exp3"] = E.verdict_exp3(E.exp3(s.delta, s.k, s.arl0))
    out["exp4"] = E.verdict_exp4(E.exp4(s.ln_delta, s.ln_gamma))
    return out


def test_preset_standard_der_cusum_ist_schneller_umschlag_2_4375_k_ist_optimal():
    v = _lines(STANDARD)
    assert v["exp1"][0] == "success" and "**8,573**" in v["exp1"][1] and "**26,746**" in v["exp1"][1]
    assert "Ab δ = 2,4375" in v["exp2"][1] and "das 0,321fache" in v["exp2"][1]
    assert "das **1,000fache**" in v["exp3"][1]                       # k = 0,5 = δ/2: kein Preis
    assert "bei 8 von 8 geprüften γ über" in v["exp4"][1]


def test_preset_kleiner_sprung_der_cusum_ist_um_das_dreifache_schneller():
    v = _lines("Kleiner Sprung δ = 0,5")
    assert v["exp1"][0] == "success" and "**27,674**" in v["exp1"][1] and "**88,890**" in v["exp1"][1]
    assert "das **1,169fache**" in v["exp3"][1]                       # k = 0,5 bei δ = 0,5: Optimum bei k = 0,25
    assert abs(88.890 / 27.674 - 3.21) < 0.01                         # „etwa das 3,2fache“ im Hilfetext: Verhältnis der Verzögerungen Shewhart / CUSUM


def test_preset_grosser_sprung_die_shewhart_karte_ist_schneller_das_verdikt_warnt():
    v = _lines("Großer Sprung δ = 4")
    assert v["exp1"][0] == "warning" and "**1,744**" in v["exp1"][1] and "**1,126**" in v["exp1"][1]
    assert "das 1,550fache" in v["exp2"][1]
    assert "das **1,556fache**" in v["exp3"][1]


def test_preset_k_zu_gross_der_preis_ist_das_2_991fache():
    v = _lines("k zu groß (k = 1,5 bei δ = 0,5)")
    assert "ARL₁ = **70,80**" in v["exp3"][1] and "das **2,991fache**" in v["exp3"][1] and "Optimum 23,67 bei k = 0,25" in v["exp3"][1]
    assert v["exp1"][0] == "success"             # auch mit k = 1,5 ist der CUSUM hier noch schneller als die Shewhart-Karte (70,80 gegen 88,89)


def test_preset_kurze_arl0_der_umschlag_wandert_nach_links():
    v = _lines("Kurze ARL₀ = 100")
    assert "ARL₀ = 100" in v["exp2"][1] and "Ab δ = 2,0914" in v["exp2"][1]
    assert "bei 8 von 8 geprüften γ über" in v["exp4"][1] and "γ = 100" in v["exp4"][1]


def test_preset_zweiseitig_nutzt_die_eingefrorene_tabelle():
    v = _lines("Zweiseitig")
    assert v["exp2"][0] == "info" and v["exp2"][1].startswith("Zweiseitig, ARL₀ = 370, k = 0,5, h = 4,7738")
    assert "das 0,226fache" in v["exp2"][1] and "Ab δ = 2,6125" in v["exp2"][1]
    assert v["exp1"][0] == "success" and "einseitig" in v["exp1"][1]       # Experiment 1 bleibt einseitig und sagt es


def test_preset_lorden_delta_2_die_naeherung_liegt_bei_gamma_10_unter_j():
    v = _lines("Lorden: δ = 2")
    assert "bei 6 von 8 geprüften γ über" in v["exp4"][1] and "bei 2 darunter" in v["exp4"][1] and "Differenz 0,157" in v["exp4"][1]
    assert "das **1,000fache**" in v["exp3"][1]                       # k = 1 = δ/2
