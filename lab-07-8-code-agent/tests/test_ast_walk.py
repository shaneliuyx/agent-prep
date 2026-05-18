"""Offline tests for the AST walk + 4-class testability filter."""
from src.ast_walk import extract


def test_pure_stateless_function():
    src = """
def add(a: int, b: int) -> int:
    return a + b
"""
    fns = extract(src)
    assert len(fns) == 1
    assert fns[0].name == "add"
    assert fns[0].testability == "pure_stateless"


def test_io_function_flagged_mock_required():
    src = """
import requests
def fetch_user(uid: int) -> dict:
    return requests.get(f"/u/{uid}").json()
"""
    fns = extract(src)
    assert fns[0].testability == "mock_required"


def test_decorated_function_flagged_fixture_required():
    src = """
def app_route(path):
    def decorator(fn): return fn
    return decorator

@app_route("/users")
def user_handler():
    return "users"
"""
    fns = extract(src)
    user_handler = next(f for f in fns if f.name == "user_handler")
    assert user_handler.testability == "fixture_required"


def test_dynamic_dispatch_flagged_escalate():
    src = """
def call_method(obj, method_name: str, *args):
    return getattr(obj, method_name)(*args)
"""
    fns = extract(src)
    assert fns[0].testability == "escalate"


def test_concurrency_flagged_property_test():
    src = """
import threading
def producer(q):
    lock = threading.Lock()
    with lock:
        q.put(1)
"""
    fns = extract(src)
    assert fns[0].testability == "property_test"


def test_nested_function_skipped():
    """Nested defs have col_offset > 0 and should be excluded."""
    src = """
def outer():
    def inner():
        pass
    return inner
"""
    fns = extract(src)
    names = {f.name for f in fns}
    assert "outer" in names
    assert "inner" not in names
