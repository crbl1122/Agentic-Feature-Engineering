"""
Unit tests for execution nodes — no LLM, uses real pandas + temp parquet.
"""
import os
import pandas as pd
import pytest

from feature_engineer.state import FeaturePlan
from feature_engineer.storage.parquet import df_to_path
from feature_engineer.nodes.execution import validate_code, create_feature, validate


@pytest.fixture
def sample_df(tmp_path):
    df = pd.DataFrame({
        "order_id":    range(10),
        "unit_price":  [10.0] * 10,
        "quantity":    [2] * 10,
        "discount_pct": [5] * 10,
        "region":      ["North"] * 5 + ["South"] * 5,
    })
    path = df_to_path(df, "test-thread")
    yield path
    if os.path.exists(path):
        os.remove(path)


def make_state(df_path, plan, errors=None, attempts=0):
    return {
        "df":       df_path,
        "plan":     plan,
        "errors":   errors or [],
        "attempts": attempts,
    }


# ── validate_code ────────────────────────────────────────────────────────────────

def test_validate_code_safe(sample_df):
    plan  = FeaturePlan(feature_name="revenue", description="x",
                        pandas_code="df['unit_price'] * df['quantity']")
    state = make_state(sample_df, plan)
    result = validate_code(state)
    assert result["errors"] == []


def test_validate_code_blocks_open(sample_df):
    plan  = FeaturePlan(feature_name="leak", description="x",
                        pandas_code="open('/etc/passwd').read()")
    state = make_state(sample_df, plan)
    result = validate_code(state)
    assert len(result["errors"]) > 0


def test_validate_code_blocks_negative_shift(sample_df):
    plan  = FeaturePlan(feature_name="leak", description="x",
                        pandas_code="df['unit_price'].shift(-1)")
    state = make_state(sample_df, plan)
    result = validate_code(state)
    assert len(result["errors"]) > 0


# ── create_feature ───────────────────────────────────────────────────────────────

def test_create_feature_success(sample_df):
    plan  = FeaturePlan(feature_name="revenue", description="Revenue",
                        pandas_code="df['unit_price'] * df['quantity']")
    state = make_state(sample_df, plan)
    result = create_feature(state)
    assert result == {}
    df = pd.read_parquet(sample_df)
    assert "revenue" in df.columns
    assert (df["revenue"] == 20.0).all()


def test_create_feature_error(sample_df):
    plan  = FeaturePlan(feature_name="bad", description="x",
                        pandas_code="df['nonexistent_column'] * 2")
    state = make_state(sample_df, plan)
    result = create_feature(state)
    assert len(result["errors"]) > 0


# ── validate ─────────────────────────────────────────────────────────────────────

def test_validate_passes(sample_df):
    plan  = FeaturePlan(feature_name="revenue", description="x",
                        pandas_code="df['unit_price'] * df['quantity']")
    # first create the column
    create_feature(make_state(sample_df, plan))
    result = validate(make_state(sample_df, plan))
    assert result["errors"] == []


def test_validate_constant_fails(sample_df):
    plan  = FeaturePlan(feature_name="constant_col", description="x",
                        pandas_code="df['unit_price'] * 0")
    create_feature(make_state(sample_df, plan))
    result = validate(make_state(sample_df, plan))
    assert any("constant" in e for e in result["errors"])
