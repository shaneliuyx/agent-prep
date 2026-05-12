# tests/conftest.py
"""Bootstrap so tests/ can import src/ as a sibling package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
