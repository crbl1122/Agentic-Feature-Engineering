"""
Derives snake_case feature names from pandas expressions via a small LLM call.
Keeping this separate makes it easy to mock in tests.
"""
from feature_engineer.llm.setup import llm


def derive_feature_name(pandas_code: str) -> str:
    """Ask LLM to derive a snake_case column name from a pandas expression."""
    prompt = (
        f"Given this pandas expression:\n  {pandas_code}\n\n"
        "Return ONLY a short snake_case column name that describes what it computes. "
        "No explanation, no punctuation, just the name.\n"
        "Examples:\n"
        "  pd.to_datetime(df['date']).dt.dayofweek → order_day_of_week\n"
        "  df['unit_price'] * df['quantity']       → total_revenue\n"
        "  df.groupby('region')['qty'].transform('sum') → total_qty_per_region\n"
    )
    return llm.invoke(prompt).content.strip().lower().replace(" ", "_")
