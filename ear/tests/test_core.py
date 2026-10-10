import numpy as np
import pytest

from core.cache import source_key
from core.context import Context
from core.pipeline import Fatal, Stage, StageUnavailable, run_pipeline
from core.timefmt import fmt_time, parse_time


def _stage(name, calls, version=1, deps=(), enabled=lambda c: True, fn=None):
    def run(ctx):
        calls.append(name)
        if fn:
            return fn(ctx)
        return {"value": name}

    return Stage(name, version, deps, run, enabled)


def test_parse_and_format_time():
    assert parse_time("75") == 75.0
    assert parse_time("1:15") == 75.0
    assert parse_time("1:01:15.5") == 3675.5
    assert fmt_time(30.97, precise=True) == "0:30.97"
    assert fmt_time(75) == "1:15"
    assert fmt_time(3675) == "1:01:15"
    with pytest.raises(ValueError):
        parse_time("1:2:3:4")


def test_source_key_depends_on_content_and_range(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    a.write_bytes(b"one")
    b.write_bytes(b"one")
    assert source_key(a, None, None) == source_key(b, None, None)
    assert source_key(a, 1.0, 5.0) != source_key(a, None, None)
    b.write_bytes(b"two")
    assert source_key(a, None, None) != source_key(b, None, None)


def test_pipeline_caches_and_reruns_dependants(tmp_path):
    calls = []
    ctx = Context(workdir=tmp_path)
    stages = [_stage("a", calls), _stage("b", calls, deps=("a",))]
    assert run_pipeline(ctx, stages) == {"a": "ok", "b": "ok"}
    assert run_pipeline(ctx, stages) == {"a": "cached", "b": "cached"}
    stages[0] = _stage("a", calls, version=2)
    assert run_pipeline(ctx, stages) == {"a": "ok", "b": "ok"}
    assert calls == ["a", "b", "a", "b"]


def test_force_and_skip(tmp_path):
    calls = []
    ctx = Context(workdir=tmp_path)
    stages = [_stage("a", calls), _stage("b", calls, enabled=lambda c: False)]
    assert run_pipeline(ctx, stages) == {"a": "ok", "b": "skipped"}
    assert run_pipeline(ctx, stages, force={"a"}) == {"a": "ok", "b": "skipped"}
    assert calls == ["a", "a"]


def test_unavailable_and_unexpected_errors_do_not_stop_the_run(tmp_path):
    calls = []
    ctx = Context(workdir=tmp_path)

    def bad(ctx):
        raise StageUnavailable("no model")

    def boom(ctx):
        raise RuntimeError("kaput")

    stages = [_stage("x", calls, fn=bad), _stage("y", calls, fn=boom), _stage("z", calls)]
    status = run_pipeline(ctx, stages)
    assert status == {"x": "unavailable: no model", "y": "unavailable: RuntimeError: kaput", "z": "ok"}
    assert ctx.data("x") is None
    assert ctx.read("x")["_unavailable"] == "no model"
    # unavailable stages retry on the next run
    assert run_pipeline(ctx, stages)["x"] == "unavailable: no model"
    assert calls.count("x") == 2


def test_fatal_stops_the_run(tmp_path):
    calls = []

    def fatal(ctx):
        raise Fatal("no audio stream")

    with pytest.raises(Fatal):
        run_pipeline(Context(workdir=tmp_path), [_stage("a", calls, fn=fatal), _stage("b", calls)])
    assert calls == ["a"]


def test_corrupt_stage_json_is_treated_as_missing(tmp_path):
    (tmp_path / "a.json").write_text('{"_v": 1, "val')
    calls = []
    assert run_pipeline(Context(workdir=tmp_path), [_stage("a", calls)]) == {"a": "ok"}
    assert calls == ["a"]


def test_write_is_atomic_and_handles_numpy(tmp_path):
    ctx = Context(workdir=tmp_path)
    ctx.write("x", {"a": np.float32(1.5), "b": np.arange(3), "c": np.bool_(True)})
    assert ctx.read("x") == {"a": 1.5, "b": [0, 1, 2], "c": True}
    assert not list(tmp_path.glob("*.tmp"))
