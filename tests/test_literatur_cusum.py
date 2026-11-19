"""Literaturwerte (nur Quellen, die im Wortlaut gelesen sind, Stufe W), gegen die 2D-Kette des zweiseitigen CUSUM (Band 0,5 %).

* Lucas, Saccucci 1990, Technometrics 32, Tab. 6 (Stufe W laut Quellenbericht; die Spaltenzuordnung der OCR-Tabelle ist nur durch Rechnung bestätigt): zweiseitiger CUSUM mit
  k = 0,5 und h = 5, zero-state: ARL0 465; ARL1 38,0 / 10,4 / 5,75 / 4,01 bei delta = 0,5 / 1 / 1,5 / 2. Stand der Zahlen: PLAN-02 und QUELLEN_KLASSISCH (2026-11-18).
* Zwetsloot, Woodall 2017 (Stufe W für Zusammenfassung und Tabellenzeile): H = 4,17 für ARL0 = 200 (k = 0,5, zweiseitig; berechnet mit dem R-Paket spc; H ist auf zwei Stellen
  gerundet, daher das Band).
NICHT geprüft, weil nicht gelesen: Page 1954, Lorden 1971 (nur Abstract), Moustakides 1986, Lucas 1982, Brook/Evans 1972. Keine Zahlen aus NIST-Tabellen und keine SADD-Spalten
von Moustakides u. a. 2009 (zu klein um 0,27 bzw. 0,51, Vormessung V1). Die ARL0-Spalten von Moustakides u. a. 2009 (Tab. 3 und 4) sind hier NICHT getestet: ihre Zahlen
liegen dem Bau nicht im Wortlaut vor. Modell: unabhängige Normalwerte, Ziel und sigma bekannt; Zähleinheit Beobachtungen."""

import pytest

import cus_arl as A


@pytest.mark.parametrize(("delta", "tabelle"), [(0.0, 465), (0.5, 38.0), (1.0, 10.4), (1.5, 5.75), (2.0, 4.01)])
def test_lucas_saccucci_1990_tab_6_cusum_h5_k05_zweiseitig(delta, tabelle):
    assert A.grid2_arl_richardson(0.5, 5.0, delta) == pytest.approx(tabelle, rel=5e-3)


def test_zwetsloot_woodall_2017_h_417_fuer_arl0_200():
    assert A.grid2_arl_richardson(0.5, 4.17, 0.0) == pytest.approx(200, rel=5e-3)
    # die Shewhart-Grenze derselben Quelle (zweiseitig, ARL0 = 200): c = 2,807
    assert A.shewhart_c(200, two=True) == pytest.approx(2.807, abs=5e-4)
