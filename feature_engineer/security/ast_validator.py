"""
AST-based security validator for LLM-generated pandas expressions.
No external dependencies — pure Python ast module.
"""
import ast

_BLOCKED_CALLS = frozenset({
    "exec", "eval", "compile", "open", "__import__",
    "breakpoint", "input", "print", "vars", "dir",
})


def assert_safe(code: str) -> None:
    """
    Parse the expression as an AST and reject:
    - dunder attribute access  (__class__, __subclasses__, ...)
    - import statements
    - calls to known dangerous builtins
    - shift(-N) — deterministic temporal leakage

    Raises ValueError with a human-readable message on violation.
    """
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in generated code: {e}") from e

    for node in ast.walk(tree):
        # block __dunder__ attribute access
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(
                f"Unsafe attribute access '{node.attr}' blocked by AST validator."
            )

        # block import statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed in pandas expressions.")

        if isinstance(node, ast.Call):
            # block dangerous builtin calls
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
                raise ValueError(
                    f"Call to '{node.func.id}' is not allowed in pandas expressions."
                )

            # block shift(-N) — always temporal leakage
            if isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
                if node.args:
                    arg = node.args[0]
                    is_negative = (
                        isinstance(arg, ast.UnaryOp)
                        and isinstance(arg.op, ast.USub)
                    ) or (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, (int, float))
                        and arg.value < 0
                    )
                    if is_negative:
                        raise ValueError(
                            "Temporal leakage: shift(-N) uses future rows — "
                            "blocked by AST validator."
                        )
