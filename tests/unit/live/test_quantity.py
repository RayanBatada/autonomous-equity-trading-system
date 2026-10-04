"""Quantity coercion: the read path must be fraction-safe BEFORE the write path
can emit a fraction, or enabling fractional sizing strands positions the
reconciler can no longer see."""

import pytest

from sma.live.quantity import QTY_EPS, as_qty, fmt_qty, is_zero, qty_eq, truncate_qty


@pytest.mark.parametrize("raw", ["175", 175, 175.0, "175.000000000"])
def test_as_qty_returns_int_for_whole_quantities(raw):
    """Bit-identity guard: for every quantity the live book has ever held,
    as_qty must return exactly what int(float(x)) returned."""
    got = as_qty(raw)
    assert got == 175
    assert isinstance(got, int)
    assert got == int(float(raw))


@pytest.mark.parametrize(
    ("raw", "want"),
    [("2.5", 2.5), (0.709973641, 0.709973641), ("-1.25", -1.25)],
)
def test_as_qty_preserves_fractions(raw, want):
    got = as_qty(raw)
    assert got == pytest.approx(want)
    assert isinstance(got, float)


def test_as_qty_does_not_truncate_the_way_int_float_did():
    """The regression this exists to prevent: int(float("2.5")) == 2."""
    assert int(float("2.5")) == 2
    assert as_qty("2.5") == 2.5


def test_truncate_never_rounds_up():
    """Rounding up would buy more than the cash-floor check budgeted."""
    assert truncate_qty(1.9999999999, 9) == pytest.approx(1.999999999)
    assert truncate_qty(0.0000000009, 9) == pytest.approx(0.0)
    assert truncate_qty(2.0, 9) == 2.0


def test_truncate_rejects_negative_precision():
    with pytest.raises(ValueError, match="precision"):
        truncate_qty(1.0, -1)


def test_qty_eq_survives_float_arithmetic():
    """A full exit is detected as delta == -held. Bare == turns a liquidation
    into a partial that leaves permanently un-exitable dust."""
    assert 0.1 + 0.2 != 0.3
    assert qty_eq(0.1 + 0.2, 0.3)
    assert not qty_eq(2.5, 2.4)


def test_is_zero():
    assert is_zero(0.0)
    assert is_zero(QTY_EPS / 2)
    assert not is_zero(1e-9)


def test_fmt_qty_leaves_whole_numbers_looking_whole():
    assert fmt_qty(175) == "175"
    assert fmt_qty(175.0) == "175"
    assert fmt_qty(175, width=5) == "  175"
    assert fmt_qty(2.5) == "2.5"
    assert fmt_qty(0.709973641) == "0.709973641"
