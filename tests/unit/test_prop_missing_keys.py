"""Property 14: missing key values are counted per column by ``validate_dataframe``."""
import re

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st
from pandas.testing import assert_frame_equal

from pylindas.lindas.exceptions import InputValidationError
from pylindas.pycube.validation import validate_dataframe

NAMES = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=8)
KEY_PREFIX = "Key column"
KEY_RE = re.compile(r"^Key column '([^']*)' has (\d+) missing value\(s\)\.$")

# (kind, valid-value strategy, missing markers usable for that kind)
KINDS = {
    "object_str": (st.text(alphabet="abc", min_size=1, max_size=3), [None, float("nan"), pd.NA]),
    "object_int": (st.integers(0, 50), [None, float("nan"), pd.NA]),
    "float64": (st.floats(0, 100, allow_nan=False), [float("nan"), None]),
    "Float64": (st.floats(0, 100, allow_nan=False), [pd.NA, None, float("nan")]),
    "Int64": (st.integers(0, 50), [pd.NA, None]),
    "string": (st.text(alphabet="abc", min_size=1, max_size=3), [pd.NA, None]),
}


def _make_series(kind, values):
    if kind == "object_str" or kind == "object_int":
        return pd.Series(values, dtype=object)
    return pd.Series(values, dtype=kind)


@st.composite
def scenario(draw):
    n_rows = draw(st.integers(1, 12))
    names = draw(st.lists(NAMES, min_size=1, max_size=8, unique=True))
    n_keys = draw(st.integers(0, len(names)))
    keys = names[:n_keys]
    data, expected_missing = {}, {}
    for name in names:
        kind = draw(st.sampled_from(sorted(KINDS)))
        valid, markers = KINDS[kind]
        mask = draw(st.lists(st.booleans(), min_size=n_rows, max_size=n_rows))
        values = [
            draw(st.sampled_from(markers)) if m else draw(valid)
            for m in mask
        ]
        data[name] = _make_series(kind, values)
        expected_missing[name] = sum(mask)
    cols = draw(st.permutations(names))
    df = pd.DataFrame(data, columns=list(cols))
    return df, list(df.columns), keys, expected_missing


# Feature: lindas-publication, Property 14: Missing key values are counted per column
# **Validates: Requirements 6.5**
@settings(max_examples=200, deadline=None)
@given(scenario())
def test_missing_key_values_counted_per_column(sc):
    df, expected, keys, expected_missing = sc
    before = df.copy(deep=True)

    problems = []
    try:
        validate_dataframe(df, expected, keys)
    except InputValidationError as exc:
        problems = list(exc.problems)

    reported = {}
    for p in problems:
        if p.startswith(KEY_PREFIX):
            m = KEY_RE.match(p)
            assert m, p
            assert m.group(1) not in reported
            reported[m.group(1)] = int(m.group(2))

    # Exactly the key columns with >= 1 missing value, with exact counts.
    assert reported == {k: expected_missing[k] for k in keys if expected_missing[k] > 0}
    # Non-key columns never appear as missing-key problems.
    assert not (set(reported) - set(keys))
    # Input frame is not mutated.
    assert_frame_equal(df, before)
    # No missing-key problems means validation passes unless duplicates are the cause.
    if not reported:
        assert all(not p.startswith(KEY_PREFIX) for p in problems)
