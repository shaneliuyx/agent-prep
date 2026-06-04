"""Put src/ on the path so flat module imports (`import observability`) resolve
under pytest. The lab runs CLI as `PYTHONPATH=src python -m <module>`."""
import pathlib
import sys

from dotenv import load_dotenv

_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
load_dotenv(_ROOT / ".env")  # real oMLX key etc. for the LLM tests
