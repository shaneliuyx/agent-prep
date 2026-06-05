"""Put the lab's `src/` on sys.path so tests can `import resumable_ingest` etc.
(the lab's modules import each other by bare name, e.g. `from ingest_agent import`)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
