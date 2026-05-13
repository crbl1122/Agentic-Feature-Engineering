"""
Unit tests for AST validator — no LLM, no external dependencies.
"""
import pytest
from feature_engineer.security.ast_validator import assert_safe


# ── Should PASS ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", [
    "df['unit_price'] * df['quantity']",
    "df.groupby('region')['unit_price'].transform('sum')",
    "pd.to_datetime(df['date']).dt.dayofweek",
    "(df['unit_price'] * (1 - df['discount_pct'] / 100))",
    "df['quantity'].shift(1)",   # positive shift is safe
    "df['quantity'].shift(3)",
])
def test_safe_expressions(code):
    assert_safe(code)  # should not raise


# ── Should FAIL — security ───────────────────────────────────────────────────────

def test_blocks_open():
    with pytest.raises(ValueError, match="open"):
        assert_safe("open('/etc/passwd').read()")


def test_blocks_exec():
    with pytest.raises(ValueError, match="exec"):
        assert_safe("exec('import os')")


def test_blocks_dunder():
    with pytest.raises(ValueError, match="__class__"):
        assert_safe("().__class__.__subclasses__()")


def test_blocks_import():
    with pytest.raises(ValueError, match="Import"):
        assert_safe("import os")


# ── Should FAIL — temporal leakage ──────────────────────────────────────────────

def test_blocks_negative_shift():
    with pytest.raises(ValueError, match="shift"):
        assert_safe("df['date'].shift(-1)")


def test_blocks_negative_shift_unary():
    with pytest.raises(ValueError, match="shift"):
        assert_safe("df['date'].shift(-3)")


def test_blocks_syntax_error():
    with pytest.raises(ValueError, match="Syntax"):
        assert_safe("df['a'] *&* df['b']")
