"""Pure DataFrame validation for ``Cube`` inputs (Requirement 6). No network, no mutation."""
from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from pylindas.lindas.exceptions import InputValidationError

MAX_LISTED_DUPLICATES = 10


def dimension_columns(cube_yaml: Mapping) -> list[str]:
    """Unique column names of the description, in order of first appearance.

    Walks ``dimensions`` first, then ``measures`` (if present). A description normally
    keeps measures inside ``dimensions`` (``dimension-type: Measure Dimension``); the
    ``measures`` walk is for descriptions that list them separately.
    """
    names: list[str] = []
    for section in ("dimensions", "measures"):
        entries = cube_yaml.get(section) or {}
        for name in entries:
            if name not in names:
                names.append(name)
    return names


def _fmt(values) -> str:
    return ", ".join(repr(str(v)) for v in values)


def validate_dataframe(df: pd.DataFrame, expected: Sequence[str], key_columns: Sequence[str]) -> None:
    """Check ``df`` against the expected columns and key columns.

    Collects all problems and raises a single ``InputValidationError``. ``df`` is only read.
    """
    problems: list[str] = []
    actual = set(df.columns)
    expected_set = set(expected)

    extra = sorted(actual - expected_set, key=str)
    if extra:
        problems.append(f"Unexpected columns not in the cube description: {_fmt(extra)}")
    missing = sorted(expected_set - actual, key=str)
    if missing:
        problems.append(f"Columns required by the cube description are missing: {_fmt(missing)}")

    if len(df) == 0:
        problems.append("The DataFrame has no rows.")

    present_keys = [k for k in dict.fromkeys(key_columns) if k in actual]

    for col in present_keys:
        n = int(df[col].isna().sum())
        if n:
            problems.append(f"Key column {str(col)!r} has {n} missing value(s).")

    if present_keys and len(present_keys) == len(set(key_columns)) and len(df) > 0:
        dup_mask = df.duplicated(subset=present_keys, keep=False)
        if dup_mask.any():
            combos = list(dict.fromkeys(map(tuple, df.loc[dup_mask, present_keys].itertuples(index=False, name=None))))
            listed = combos[:MAX_LISTED_DUPLICATES]
            shown = "; ".join(
                "(" + ", ".join(f"{c}={v!r}" for c, v in zip(present_keys, combo)) + ")" for combo in listed
            )
            problems.append(
                f"{len(combos)} duplicate key combination(s) over {_fmt(present_keys)}"
                f" (showing {len(listed)}): {shown}"
            )

    if problems:
        raise InputValidationError(problems)
