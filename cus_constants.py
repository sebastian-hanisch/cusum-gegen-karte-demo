"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Modell (überall): unabhängige Beobachtungen x_t, N(0, 1) vor und N(delta, 1) ab der Änderung; Ziel 0 und sigma = 1 bekannt.
nu = Zahl der Beobachtungen vor der Änderung. Zähleinheit: Beobachtungen bis zum Alarm (die Alarmbeobachtung zählt mit)."""

# --- Regler (Seitenleiste) -------------------------------------------------------------------------------------------------------------------
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.25, 4.0, 0.25, 1.0
K_MIN, K_MAX, K_STEP, DEFAULT_K = 0.05, 2.5, 0.025, 0.5
ARL0_OPTIONS = (100, 200, 370, 500, 1000, 2000, 5000)
DEFAULT_ARL0 = 370
NU_MIN, NU_MAX, DEFAULT_NU = 0, 200, 50
SEED_MAX, DEFAULT_SEED = 999999, 35
R_MIN, R_MAX, R_STEP, DEFAULT_R = 50, 500, 50, 300
ZAEHLUNG = ("einseitig", "zweiseitig")
MASS = ("ARL1 in Beobachtungen", "Verhältnis zur Shewhart-Karte")
LN_DELTAS = (0.5, 1.0, 2.0)
LN_GAMMAS = (10, 30, 100, 370, 1000, 10000, 100000, 1000000)
DEFAULT_LN_DELTA, DEFAULT_LN_GAMMA = 1.0, 370

# --- feste Annahmen ----------------------------------------------------------------------------------------------------------------------------
SERIES_LENGTH = 250                 # Beobachtungen der gezeigten Einzelreihe
FALSE_ALARM_HORIZON = 100           # „Fehlalarm vor Beobachtung 100“ (Anteil der Reihen ohne Sprung mit Alarm)
SIM_BUDGET = 1_000_000              # Normalzahlen je Simulation (R Reihen mit Sprung plus R Reihen ohne): Obergrenze für R; rund 0,5 bis 1,2 s (Messung 2026-11-18)
SIM_TCAP = 200_000                  # Notbremse je Reihe (kein Alarm in so vielen Beobachtungen)
GRID_M = 300                        # Zellen des Brook/Evans-Gitters für die exakte Lauflängenverteilung der App
NODES = 120                         # Knoten der Integralgleichung (200 ab ARL0 = 10^4 in der Lorden-Tabelle)
ENVELOPE_DELTAS = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)   # Raster der ARL1-Kurve über delta
TABLE_DELTAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)        # Spalten der Vergleichs- und k-Tabellen
TABLE_KS = (0.25, 0.5, 0.75, 1.0, 1.5)                            # Zeilen der k-Tabelle
TWOSIDED_KS = (0.25, 0.5, 0.75, 1.0)                              # eingefrorene zweiseitige Tabelle: k
TWOSIDED_ARL0 = (100, 370, 1000)                                  # ... und ARL0
TWOSIDED_DELTAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
HYBRID_CS = (None, 4.0, 3.5, 3.0)                                 # None = reiner CUSUM (c = unendlich)
HYBRID_DELTAS = (0.5, 1.0, 2.0, 3.0, 4.0)

# --- Farben ------------------------------------------------------------------------------------------------------------------------------------
COLOR_SHEWHART = "#d95f02"
COLOR_CUSUM = "#2F6B65"
COLOR_ENVELOPE = "#4c78a8"
COLOR_GREY = "rgba(150,150,150,0.55)"

# --- Presets -----------------------------------------------------------------------------------------------------------------------------------
_BASIS = {
    "delta": DEFAULT_DELTA, "arl0": DEFAULT_ARL0, "k": DEFAULT_K, "nu": DEFAULT_NU, "seed": DEFAULT_SEED, "reihen": DEFAULT_R,
    "zaehlung": ZAEHLUNG[0], "mass": MASS[0], "ln_delta": DEFAULT_LN_DELTA, "ln_gamma": DEFAULT_LN_GAMMA,
}
PRESETS = {
    "Standard: Sprung δ = 1, ARL₀ = 370": dict(_BASIS),
    "Kleiner Sprung δ = 0,5": {**_BASIS, "delta": 0.5, "ln_delta": 0.5},
    "Großer Sprung δ = 4": {**_BASIS, "delta": 4.0},
    "k zu groß (k = 1,5 bei δ = 0,5)": {**_BASIS, "delta": 0.5, "k": 1.5},
    "Kurze ARL₀ = 100": {**_BASIS, "arl0": 100, "ln_gamma": 100},
    "Zweiseitig": {**_BASIS, "zaehlung": ZAEHLUNG[1]},
    "Lorden: δ = 2": {**_BASIS, "delta": 2.0, "k": 1.0, "ln_delta": 2.0, "ln_gamma": 10},
}
PRESET_HELP = {
    "Standard: Sprung δ = 1, ARL₀ = 370": "Sprung δ = 1 (eine Standardabweichung); beide Karten melden ohne Sprung im Mittel nach 370 Beobachtungen falsch; CUSUM mit k = 0,5, einseitig.",
    "Kleiner Sprung δ = 0,5": "Halbe Standardabweichung: die Verzögerung der Shewhart-Karte ist hier das 3,2fache der des CUSUM (k = 0,5, ARL₀ = 370, einseitig).",
    "Großer Sprung δ = 4": "Vier Standardabweichungen: der CUSUM mit k = 0,5 braucht das 1,55fache der Shewhart-Verzögerung (der Umschlag liegt bei δ = 2,4375).",
    "k zu groß (k = 1,5 bei δ = 0,5)": "k = 1,5 passt zu δ = 3, der wahre Sprung ist 0,5: die Verzögerung ist das 2,99fache der besten (ARL₁ 70,80 statt 23,67 bei k = 0,25).",
    "Kurze ARL₀ = 100": "Beide Karten melden ohne Sprung im Mittel nach 100 Beobachtungen falsch: der Umschlag wandert nach links, δ = 2,09 statt 2,44 (k = 0,5, einseitig).",
    "Zweiseitig": "Beide Richtungen werden überwacht. Zweiseitig gilt h = 4,7738 und c = 2,9997 für ARL₀ = 370; das Verhältnis zur Shewhart-Karte hängt am Zählen (δ = 1: 0,226 zweiseitig, 0,321 einseitig).",
    "Lorden: δ = 2": "Bei δ = 2 und γ = 10 liegt die Näherung ln γ / I unter der exakten Verzögerung (1,1513 gegen 1,3083); bei δ = 1 liegt sie bei allen acht geprüften γ darüber.",
}
