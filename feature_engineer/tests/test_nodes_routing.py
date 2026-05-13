"""
Unit tests for routing functions — pure functions, no LLM, no I/O.
"""
import pytest
from feature_engineer.nodes.routing import (
    should_execute, should_retry, after_record, after_next,
)


def make_state(**kwargs):
    base = {
        "plan":            None,
        "feature_queue":   [],
        "errors":          [],
        "attempts":        0,
        "completed_features": [],
    }
    base.update(kwargs)
    return base


class _MockPlan:
    feature_name = "test_feature"


# ── should_execute ───────────────────────────────────────────────────────────────

def test_should_execute_run():
    assert should_execute(make_state(plan=_MockPlan())) == "run"


def test_should_execute_next_when_queue():
    state = make_state(plan=_MockPlan(), errors=["blocked"],
                       feature_queue=[{"feature_name": "f2", "pandas_code": "x"}])
    assert should_execute(state) == "next"


def test_should_execute_save_when_no_queue():
    state = make_state(plan=_MockPlan(), errors=["blocked"], feature_queue=[])
    assert should_execute(state) == "save"


# ── should_retry ─────────────────────────────────────────────────────────────────

def test_should_retry_revise():
    state = make_state(plan=_MockPlan(), errors=["fail"], attempts=0)
    assert should_retry(state) == "revise"


def test_should_retry_record():
    state = make_state(plan=_MockPlan(), errors=[])
    assert should_retry(state) == "record"


def test_should_retry_skip_to_next():
    state = make_state(plan=_MockPlan(), errors=["fail"], attempts=3,
                       feature_queue=[{"feature_name": "f2", "pandas_code": "x"}])
    assert should_retry(state) == "next"


def test_should_retry_skip_to_save():
    state = make_state(plan=_MockPlan(), errors=["fail"], attempts=3, feature_queue=[])
    assert should_retry(state) == "save"


# ── after_record ─────────────────────────────────────────────────────────────────

def test_after_record_next():
    state = make_state(feature_queue=[{"feature_name": "f2", "pandas_code": "x"}])
    assert after_record(state) == "next"


def test_after_record_save():
    assert after_record(make_state()) == "save"


# ── after_next ───────────────────────────────────────────────────────────────────

def test_after_next_run():
    assert after_next(make_state()) == "run"


def test_after_next_save_exhausted():
    state = make_state(errors=["__queue_exhausted__"])
    assert after_next(state) == "save"


def test_after_next_revise_duplicate():
    state = make_state(errors=["__duplicate__: 'x' already completed"])
    assert after_next(state) == "revise"
