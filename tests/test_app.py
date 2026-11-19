"""End-to-end-Rauchtest über Streamlits AppTest: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Regler an seine Grenzen (Eckfall-Matrix, DEMO-PLAYBOOK §1d)
und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei Diagrammen mit gleichem Inhalt und Slider mit berechneten Grenzen).

Dazu die Entartungen dieser Demo: nicht abstimmbares k, zweiseitig außerhalb der Tabelle, Simulationsbudget bei langer Shewhart-Verzögerung, ν größer als die Reihenlänge."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import cus_constants as C
import cus_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
REGLER_ARTEN = ("slider", "number_input", "select_slider", "radio")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def _kante(spec, kante):
    if spec.options is not None:
        return spec.options[0] if kante == "min" else spec.options[-1]
    return spec.lo if kante == "min" else spec.hi


def _setze(at, werte):
    for key, wert in werte.items():
        _regler(at, key).set_value(wert)
    at.run(timeout=TIMEOUT)
    assert not at.exception, (werte, [str(e) for e in at.exception])
    return at


def test_app_laeuft_ohne_ausnahme_und_zeigt_die_vier_experimente():
    at = _fresh_app()
    titel = [s.value for s in at.subheader]
    assert [t for t in titel if t.startswith("🔬 Experiment")] == [
        "🔬 Experiment 1: viele Reihen gegen die exakte Verteilung", "🔬 Experiment 2: gleiche ARL₀, wer ist schneller, und ab wann kippt es?",
        "🔬 Experiment 3: welches k? Das Referenzmaß ist eine Annahme", "🔬 Experiment 4: wie schnell könnte man höchstens sein? Die Näherung ln γ / I"]
    assert any(s.value == "🚧 Wo die Annahmen enden" for s in at.subheader)
    labels = [e.label for e in at.expander]
    assert any(x == "📐 Mathematische Formulierung" for x in labels) and any("So funktioniert der CUSUM" in x for x in labels)
    assert not at.error


def test_jeder_button_klick_loest_keine_ausnahme_aus():
    labels = [b.label for b in _fresh_app().sidebar.button]
    assert len(labels) == len(C.PRESETS) + 2
    for index in range(len(labels)):
        at = _fresh_app()
        at.sidebar.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (labels[index], [str(e) for e in at.exception])


@pytest.mark.parametrize("key", list(P.SETTING_SPECS))
@pytest.mark.parametrize("kante", ["min", "max"])
def test_jeder_regler_an_seiner_grenze_loest_keine_ausnahme_aus(key, kante):
    at = _fresh_app()
    _setze(at, {key: _kante(P.SETTING_SPECS[key], kante)})


def test_alle_regler_auf_minimum_und_alle_auf_maximum():
    for kante in ("min", "max"):
        _setze(_fresh_app(), {key: _kante(spec, kante) for key, spec in P.SETTING_SPECS.items()})


@pytest.mark.parametrize("werte", [
    {"k_slider": 2.5, "arl0_select": 100},                                              # k nicht abstimmbar (ARL0 = 100 erlaubt k nur bis 2,326)
    {"zaehlung_radio": "zweiseitig", "k_slider": 1.5},                                  # zweiseitig außerhalb der eingefrorenen Tabelle
    {"zaehlung_radio": "zweiseitig", "arl0_select": 200},
    {"delta_slider": 0.25, "arl0_select": 5000, "reihen_slider": 500},                  # Rechenbudget kappt die Zahl der Reihen
    {"nu_slider": 200, "delta_slider": 0.25},                                           # kaum ein Alarm in 250 Beobachtungen
    {"mass_radio": C.MASS[1], "zaehlung_radio": "zweiseitig"},
    {"mass_radio": C.MASS[1], "delta_slider": 4.0, "k_slider": 0.05},
    {"ln_delta_select": 2.0, "ln_gamma_select": 1000000},
])
def test_eckfaelle_der_demo(werte):
    at = _setze(_fresh_app(), werte)
    if werte.get("k_slider") == 2.5 and werte.get("arl0_select") == 100:
        assert any("nicht erreichbar" in w.value for w in at.warning)
    if werte.get("zaehlung_radio") == "zweiseitig" and (werte.get("k_slider") == 1.5 or werte.get("arl0_select") == 200):
        assert any("eingefroren" in w.value for w in at.warning)
    if werte.get("reihen_slider") == 500 and werte.get("arl0_select") == 5000:
        assert any("gekappt" in i.value for i in at.info)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_setzt_die_regler_und_laeuft(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == C.PRESETS[name][key], (name, key)


def test_permalink_uebernimmt_werte_und_begrenzt_sie():
    at = AppTest.from_file(APP_PATH)
    at.query_params["delta"] = "99"
    at.query_params["arl0"] = "500"
    at.query_params["z"] = "zweiseitig"
    at.query_params["lng"] = "7"          # keine erlaubte Option: wird ignoriert
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.session_state["delta_slider"] == C.DELTA_MAX and at.session_state["arl0_select"] == 500
    assert at.session_state["zaehlung_radio"] == "zweiseitig" and at.session_state["ln_gamma_select"] == C.DEFAULT_LN_GAMMA
