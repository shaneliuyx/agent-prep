"""Load numeric lab scripts like 02_pipeline.py or 03_hyde.py."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path


def load(script_name: str):
    path = Path(__file__).with_name(script_name)
    spec = spec_from_file_location(script_name.replace(".py", ""), path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
