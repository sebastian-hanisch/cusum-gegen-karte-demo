"""Lädt die eingefrorenen Tabellen (cus_twosided.json, cus_hybrid.json; Quelle, Stand und Verfahren stehen im Feld `_meta`, erzeugt von tools/cus_make_tables.py).

Zweiseitig gibt es nur Zahlen für k in cus_constants.TWOSIDED_KS und ARL0 in cus_constants.TWOSIDED_ARL0; sonst liefern die Funktionen None
(die App zeigt dann eine Meldung statt einer Zahl)."""

import functools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=None)
def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def twosided_cell(k: float, arl0: float):
    """Zelle der zweiseitigen Tabelle für (k, ARL0) oder None, wenn sie nicht eingefroren ist."""
    for cell in load("cus_twosided.json")["zellen"]:
        if math.isclose(cell["k"], k, abs_tol=1e-9) and cell["arl0"] == arl0:
            return cell
    return None


def twosided_deltas() -> list:
    return load("cus_twosided.json")["deltas"]


def hybrid_table() -> dict:
    return load("cus_hybrid.json")
