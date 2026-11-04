import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
KIT = REPO / "examples" / "frozen_condition_cache"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(KIT))
