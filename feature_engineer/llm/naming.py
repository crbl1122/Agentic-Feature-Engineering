"""
Derives snake_case feature names from pandas expressions via a small LLM call.
Accepts optional domain context and list of already-used names to avoid collisions.
"""
from feature_engineer.llm.setup import llm


def derive_feature_name(
    pandas_code: str,
    context: str = "",
    used_names: list[str] | None = None,
) -> str:
    """Ask LLM to derive a snake_case column name from a pandas expression.

    Args:
        pandas_code:  the pandas expression
        context:      optional domain context (e.g. objective) for domain-aware names
        used_names:   names already in use — LLM will pick a different one
    """
    context_line = f"Domain context: {context}\n" if context else ""
    used_block   = (
        f"These names are already taken — do NOT use them:\n"
        + "\n".join(f"  - {n}" for n in used_names) + "\n"
        if used_names else ""
    )
    prompt = (
        f"{context_line}"
        f"{used_block}"
        f"Given this pandas expression:\n  {pandas_code}\n\n"
        "Return ONLY a short snake_case column name that describes "
        "the PHYSICAL or DOMAIN MEANING, not the mathematical operation.\n"
        "If the expression computes a variant of an already-taken name, "
        "add a distinguishing qualifier (e.g. _transverse, _combined, _ratio).\n"
        "No explanation, no punctuation, just the name.\n\n"
        "Examples:\n"
        "  (df['Px1']+df['Px2'])**2 + (df['Py1']+df['Py2'])**2   → total_momentum_squared\n"
        "  df['E1'] + df['E2']                                     → total_energy\n"
        "  pd.to_datetime(df['date']).dt.dayofweek                 → order_day_of_week\n"
        "  df['unit_price'] * df['quantity']                       → total_revenue\n"
        "  df.groupby('region')['qty'].transform('sum')            → total_qty_per_region\n"
    )
    return llm.invoke(prompt).content.strip().lower().replace(" ", "_")
