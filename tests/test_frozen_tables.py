"""Eingefrorene Tabellen (cus_twosided.json, cus_hybrid.json): Quelle und Stand je Datei, Form, Handwerte der Vormessung und Gleichheit mit der Neurechnung.

Die Neurechnung ist das Skript tools/cus_make_tables.py (`--check` rechnet drei Zellen je Datei, `--check --lang` alle). Hier laufen drei Zellen je Datei mit relativer Toleranz 1e-8
(Gleitkomma: BLAS und SuperLU unterscheiden sich in den letzten Stellen); der Test prüft, dass die eingefrorenen Zahlen gleich der Neurechnung sind."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

import cus_arl as A
import cus_constants as C
import cus_reference as REF

ROOT = Path(__file__).resolve().parent.parent


def _tool():
    spec = importlib.util.spec_from_file_location("cus_make_tables", ROOT / "tools" / "cus_make_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cus_make_tables"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_jede_datei_nennt_quelle_stand_verfahren_und_zaehleinheit():
    for name in ("cus_twosided.json", "cus_hybrid.json"):
        meta = REF.load(name)["_meta"]
        for feld in ("quelle", "stand", "verfahren", "zaehleinheit", "modell", "python", "numpy"):
            assert meta.get(feld), (name, feld)
        assert "tools/cus_make_tables.py" in meta["quelle"] and meta["stand"] == "2026-11-18"


def test_zweiseitige_tabelle_hat_alle_zwoelf_zellen_und_neun_sprunggroessen():
    data = REF.load("cus_twosided.json")
    assert [(z["k"], z["arl0"]) for z in data["zellen"]] == [(k, a) for a in C.TWOSIDED_ARL0 for k in C.TWOSIDED_KS]
    assert data["deltas"] == list(C.TWOSIDED_DELTAS)
    for z in data["zellen"]:
        assert len(z["arl1_cusum"]) == len(z["arl1_shewhart"]) == 9
        assert z["arl0_kette"] == pytest.approx(z["arl0"], rel=2e-5)       # h ist auf die Ziel-ARL0 abgestimmt (Gitterfehler der Kette)
        assert z["umschlag"] is not None and 1.5 < z["umschlag"] < 4.0


def test_zweiseitige_handwerte_der_vormessung_k05_arl0_370():
    """V11 B: h = 4,7738 (ARL0 369,9898); ARL1 35,2534 / 9,9246 / 5,5210 / 3,8578 / 3,0005 / 2,4859 / 1,9568 bei δ = 0,5 / 1 / 1,5 / 2 / 2,5 / 3 / 4; Shewhart 155,0752 / 43,8596 / 14,9579 / 6,2997 / 3,2399 / 1,9995 / 1,1885."""
    z = REF.twosided_cell(0.5, 370)
    assert z["h"] == pytest.approx(4.7738, abs=2e-4)
    d = REF.twosided_deltas()
    cus = dict(zip(d, z["arl1_cusum"]))
    shew = dict(zip(d, z["arl1_shewhart"]))
    for delta, c_soll, s_soll in ((0.5, 35.2534, 155.0752), (1.0, 9.9246, 43.8596), (1.5, 5.5210, 14.9579), (2.0, 3.8578, 6.2997), (2.5, 3.0005, 3.2399), (3.0, 2.4859, 1.9995), (4.0, 1.9568, 1.1885)):
        assert cus[delta] == pytest.approx(c_soll, abs=1.5e-3), delta
        # die Shewhart-Grenze gehört hier zur Ziel-ARL0 = 370 (die Vormessung nahm die ARL0 369,9898 der Kette): Unterschied bis 5e-4 relativ
        assert shew[delta] == pytest.approx(s_soll, rel=6e-4), delta
    assert z["umschlag"] == pytest.approx(2.61, abs=0.01)     # Plan: Interpolation auf dem 0,1-Raster 2,61; hier brentq 2,6125


def test_hybrid_tabelle_form_und_handwerte():
    t = REF.hybrid_table()
    assert t["deltas"] == list(C.HYBRID_DELTAS) and [z["c"] for z in t["zeilen"]] == list(C.HYBRID_CS)
    erwartet = {None: (4.09548, [27.6745, 8.5731, 3.4061, 2.2307, 1.7443]), 4.0: (4.10023, [27.7032, 8.5758, 3.3849, 2.1251, 1.5199]),
                3.5: (4.14688, [27.9930, 8.6200, 3.3345, 1.9692, 1.3281]), 3.0: (4.59761, [30.9266, 9.1562, 3.3038, 1.7765, 1.1805])}
    for z in t["zeilen"]:
        h, arl1 = erwartet[z["c"]]
        assert z["h"] == pytest.approx(h, abs=5e-5)
        assert z["arl1"] == pytest.approx(arl1, abs=1.5e-3)
        assert z["arl0_gitter"] == pytest.approx(370, abs=0.05)
    assert t["shewhart_allein"] == pytest.approx([88.8904, 26.7460, 4.6049, 1.7055, 1.1256], abs=2e-3)


def test_eingefrorene_zahlen_gleich_neurechnung_je_datei_zellen():
    """Hybrid: drei Zeilen; zweiseitig: eine Zelle (k = 0,5, ARL0 = 370; das Skript selbst prüft mit `--check` drei, mit `--check --lang` alle zwölf)."""
    tool = _tool()
    assert tool.check_twosided(REF.load("cus_twosided.json"), lang=False, indices=(5,)) == []
    assert tool.check_hybrid(REF.load("cus_hybrid.json"), lang=False) == []


def test_pruefwerkzeug_erkennt_eine_verfaelschte_zelle():
    tool = _tool()
    data = {**REF.load("cus_hybrid.json")}
    zeilen = [dict(z) for z in data["zeilen"]]
    zeilen[0]["arl1"] = [x * 1.001 for x in zeilen[0]["arl1"]]
    assert tool.check_hybrid({**data, "zeilen": zeilen}, lang=False)


def test_zweiseitig_liefert_nichts_ausserhalb_der_tabelle():
    assert REF.twosided_cell(0.5, 370) is not None and REF.twosided_cell(0.3, 370) is None and REF.twosided_cell(0.5, 200) is None


def test_shewhart_zeilen_der_zweiseitigen_tabelle_sind_geschlossen_nachrechenbar():
    for z in REF.load("cus_twosided.json")["zellen"]:
        c = A.shewhart_c(z["arl0"], two=True)
        assert z["c_shewhart"] == pytest.approx(c, rel=1e-12)
        for d, s in zip(REF.twosided_deltas(), z["arl1_shewhart"]):
            assert math.isclose(s, A.shewhart_arl(c, d, two=True), rel_tol=1e-12)
