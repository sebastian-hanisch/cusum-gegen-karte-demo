"""Erzeugt und prüft die eingefrorenen Tabellen `cus_twosided.json` und `cus_hybrid.json` (schwere Rechnungen, nicht live in der App).

Aufruf (Test-venv, OMP_NUM_THREADS=1), im Demo-Ordner:
    python tools/cus_make_tables.py                 # alles neu rechnen und schreiben (rund 4 bis 6 Minuten)
    python tools/cus_make_tables.py --check         # drei Zellen je Datei neu rechnen und mit der Datei vergleichen (rund 1 Minute)
    python tools/cus_make_tables.py --check --lang  # ALLE Zellen neu rechnen und vergleichen (so lange wie das Schreiben)

Modell: unabhängige Beobachtungen N(delta, 1), Ziel 0 und sigma = 1 bekannt; Zähleinheit Beobachtungen bis zum Alarm; zero-state (= Lorden-Maß für beide Karten).
  * cus_twosided.json: zweiseitiger CUSUM (2D-Kette, Richardson aus m = 60 und 80) gegen zweiseitige Shewhart-Karte (|x| > c), je k und ARL0.
  * cus_hybrid.json:  einseitiger CUSUM mit zusätzlicher Shewhart-Grenze (Alarm bei S >= h oder x > c), k = 0,5, ARL0 = 370 (Gitter m = 320 und 640, Richardson).
"""

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

import cus_arl as A  # noqa: E402
import cus_constants as C  # noqa: E402

STAND = "2026-11-18"
TOL = 1e-8   # relative Toleranz Neurechnung gegen Datei (BLAS und SuperLU unterscheiden sich in den letzten Stellen)


def _meta(verfahren: str) -> dict:
    return {
        "quelle": "tools/cus_make_tables.py (Skript des Demo-Ordners; Rechnung im Modell der README)",
        "stand": STAND,
        "verfahren": verfahren,
        "zaehleinheit": "Beobachtungen bis zum Alarm, zero-state (= Lorden-Maß für diese Karten), Alarmbeobachtung zählt mit",
        "modell": "unabhängige Beobachtungen N(delta, 1), Ziel 0 und sigma = 1 bekannt",
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


# ----------------------------------------------------------------------------- zweiseitig
def twosided_h(k: float, arl0: float) -> float:
    """Schwelle h des zweiseitigen CUSUM mit ARL0 (Richardson aus m = 60 und 80), brentq; Start: einseitige Schwelle (h wächst zweiseitig)."""
    lo = A.cusum_tune(k, arl0)
    hi = lo * 1.3

    def f(h):
        return A.grid2_arl_richardson(k, h, 0.0) - arl0

    while f(hi) < 0:
        hi *= 1.3
    return float(brentq(f, lo, hi, xtol=1e-7))


def twosided_cell(k: float, arl0: float, h: float | None = None, deltas=C.TWOSIDED_DELTAS, with_crossover: bool = True) -> dict:
    h = twosided_h(k, arl0) if h is None else h
    c = A.shewhart_c(arl0, two=True)
    cell = {
        "k": k, "arl0": arl0, "h": h,
        "arl0_kette": A.grid2_arl_richardson(k, h, 0.0),
        "c_shewhart": c,
        "arl1_cusum": [A.grid2_arl_richardson(k, h, d) for d in deltas],
        "arl1_shewhart": [A.shewhart_arl(c, d, two=True) for d in deltas],
    }
    if with_crossover:
        d1 = A.crossover(k, arl0)  # einseitiger Umschlag als Startwert der Suche; zweiseitig liegt höher

        def diff(d):
            return A.grid2_arl_richardson(k, h, d) - A.shewhart_arl(c, d, two=True)

        lo, hi = (d1 or 2.0) - 0.2, (d1 or 2.0) + 0.9
        cell["umschlag"] = float(brentq(diff, lo, hi, xtol=1e-5)) if diff(lo) < 0 < diff(hi) else None
    return cell


def build_twosided(log=print) -> dict:
    cells = []
    for arl0 in C.TWOSIDED_ARL0:
        for k in C.TWOSIDED_KS:
            t0 = time.time()
            cells.append(twosided_cell(k, arl0))
            log(f"  zweiseitig k = {k}, ARL0 = {arl0}: h = {cells[-1]['h']:.5f}, Umschlag {cells[-1]['umschlag']}  ({time.time() - t0:.0f} s)")
    return {"_meta": _meta("zweiseitiger CUSUM als 2D-Kette (Richardson aus m = 60 und 80), zweiseitige Shewhart-Karte geschlossen"),
            "deltas": list(C.TWOSIDED_DELTAS), "zellen": cells}


# ----------------------------------------------------------------------------- Hybrid
def hybrid_cell(c: float | None, k: float = 0.5, arl0: float = 370, deltas=C.HYBRID_DELTAS) -> dict:
    cc = 1e9 if c is None else c
    h = A.hybrid_tune(k, cc, arl0)
    return {
        "c": c, "h": h, "arl0_gitter": A.hybrid_arl(k, h, 0.0, cc),
        "arl1": [A.hybrid_arl(k, h, d, cc) for d in deltas],
    }


def build_hybrid(log=print) -> dict:
    zeilen = []
    for c in C.HYBRID_CS:
        t0 = time.time()
        zeilen.append(hybrid_cell(c))
        log(f"  Hybrid c = {c}: h = {zeilen[-1]['h']:.5f}  ({time.time() - t0:.0f} s)")
    c_sh = A.shewhart_c(370)
    shew = [A.shewhart_arl(c_sh, d) for d in C.HYBRID_DELTAS]
    return {"_meta": _meta("einseitiger CUSUM (k = 0,5) mit Shewhart-Grenze c, h je c auf ARL0 = 370 abgestimmt; Gitter m = 320 und 640, Richardson"),
            "k": 0.5, "arl0": 370, "deltas": list(C.HYBRID_DELTAS), "zeilen": zeilen, "shewhart_allein": shew}


# ----------------------------------------------------------------------------- Prüfen
def _close(a, b, tol=TOL):
    return math.isclose(a, b, rel_tol=tol, abs_tol=1e-12)


def check_twosided(data: dict, lang: bool, indices=(1, 5, -1)) -> list[str]:
    fehler = []
    cells = data["zellen"]
    pick = cells if lang else [cells[i] for i in indices]
    for old in pick:
        new = twosided_cell(old["k"], old["arl0"], h=old["h"], deltas=data["deltas"], with_crossover=lang)
        for key in ("arl0_kette", "c_shewhart"):
            if not _close(old[key], new[key]):
                fehler.append(f"zweiseitig k={old['k']} ARL0={old['arl0']} {key}: Datei {old[key]} gegen neu {new[key]}")
        for key in ("arl1_cusum", "arl1_shewhart"):
            for d, a, b in zip(data["deltas"], old[key], new[key]):
                if not _close(a, b):
                    fehler.append(f"zweiseitig k={old['k']} ARL0={old['arl0']} {key} delta={d}: Datei {a} gegen neu {b}")
        if lang and old.get("umschlag") is not None and not math.isclose(old["umschlag"], new["umschlag"], abs_tol=2e-5):
            fehler.append(f"zweiseitig k={old['k']} ARL0={old['arl0']} Umschlag: Datei {old['umschlag']} gegen neu {new['umschlag']}")
    return fehler


def check_hybrid(data: dict, lang: bool, indices=(0, 2, 3)) -> list[str]:
    fehler = []
    pick = data["zeilen"] if lang else [data["zeilen"][i] for i in indices]
    for old in pick:
        cc = 1e9 if old["c"] is None else old["c"]
        h = old["h"]
        if not lang:
            new_h = A.hybrid_tune(data["k"], cc, data["arl0"])
            if not _close(h, new_h, 1e-6):
                fehler.append(f"Hybrid c={old['c']} h: Datei {h} gegen neu {new_h}")
        for d, a in zip(data["deltas"], old["arl1"]):
            b = A.hybrid_arl(data["k"], h, d, cc)
            if not _close(a, b):
                fehler.append(f"Hybrid c={old['c']} delta={d}: Datei {a} gegen neu {b}")
    return fehler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="nur prüfen (drei Zellen je Datei), nichts schreiben")
    ap.add_argument("--lang", action="store_true", help="mit --check: alle Zellen neu rechnen")
    args = ap.parse_args()
    paths = {"twosided": ROOT / "cus_twosided.json", "hybrid": ROOT / "cus_hybrid.json"}
    if args.check:
        t0 = time.time()
        fehler = []
        fehler += check_twosided(json.loads(paths["twosided"].read_text(encoding="utf-8")), args.lang)
        fehler += check_hybrid(json.loads(paths["hybrid"].read_text(encoding="utf-8")), args.lang)
        for f in fehler:
            print("FEHLER:", f)
        print(f"{'FEHLER' if fehler else 'OK'}: eingefrorene Tabellen gegen Neurechnung ({'alle Zellen' if args.lang else 'drei Zellen je Datei'}), {time.time() - t0:.0f} s")
        return 1 if fehler else 0
    t0 = time.time()
    print("Hybrid ...", flush=True)
    hyb = build_hybrid(log=lambda s: print(s, flush=True))
    paths["hybrid"].write_text(json.dumps(hyb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("zweiseitig ...", flush=True)
    two = build_twosided(log=lambda s: print(s, flush=True))
    paths["twosided"].write_text(json.dumps(two, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
