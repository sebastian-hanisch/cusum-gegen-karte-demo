"""Gemeinsame Einstellungen: Wurzelordner im Importpfad, ein BLAS-Thread (kleine dichte Lösungen sind mit mehreren Threads um den Faktor 100 langsamer)."""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
