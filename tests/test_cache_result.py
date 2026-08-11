"""Characterization tests for the `cache_result` decorator (server.py:72-96).

Known bug #1: the decorator only persists results when
`isinstance(result, (dict, list))`, but every real tool returns a Pydantic
BaseModel or list[BaseModel]. These tests use a small dummy model rather than
any live tool, and are expected to FAIL against the current implementation:
  (a) a single-BaseModel result is never cached (silent no-op cold cache).
  (b) a list-of-BaseModel result blows up with TypeError inside json.dumps().
"""
from pydantic import BaseModel

import server


class DummyModel(BaseModel):
    value: int


def test_cache_result_caches_single_basemodel_result():
    """A function returning a bare BaseModel should be served from cache on
    the second call. Currently fails: BaseModel is neither dict nor list, so
    set_cached_val is never invoked and the wrapped function re-executes."""
    calls = {"n": 0}

    @server.cache_result()
    def get_thing():
        calls["n"] += 1
        return DummyModel(value=calls["n"])

    get_thing()
    get_thing()

    assert calls["n"] == 1, "expected second call to be served from cache, not re-execute the function"


def test_cache_result_handles_list_of_basemodel_results_without_raising():
    """A function returning list[BaseModel] should cache successfully instead
    of raising. Currently fails: json.dumps() on a list of BaseModel objects
    raises TypeError (BaseModel is not JSON serializable)."""

    @server.cache_result()
    def get_things():
        return [DummyModel(value=1), DummyModel(value=2)]

    result = get_things()
    assert isinstance(result, list)
    assert len(result) == 2
