import sys
from pathlib import Path

# Ensure project root is importable when tests run from the tests directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
