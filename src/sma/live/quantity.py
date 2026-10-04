"""Share quantities that may be fractional.

Every broker read in this system used to coerce a quantity with `int(float(x))`.
That is correct exactly as long as the bot only ever SUBMITS whole shares — the
moment `sizing.fractional_shares` is on, `int(float("2.5"))` silently destroys
half a share and the ledger-vs-book drift detector reports a phantom mismatch
forever after.

`as_qty` replaces those coercions without needing the sizing flag threaded into
every module. It returns an `int` for a whole-number quantity — bit-identical to
the old `int(float(x))` for every value the live book has ever held — and a
`float` only when there is a real fraction to preserve. So the read path is
fraction-safe before the write path ever emits a fraction, and turning the flag
on cannot strand a position the reconciler can no longer see.

`QTY_EPS` is the "this is zero" tolerance. Alpaca accepts fractional quantities
to 9 decimal places, so anything below 1e-10 is either float noise from a
subtraction or a dust position that cannot legally be traded again.
"""

from __future__ import annotations

#: Quantities closer together than this are the same quantity. One order of
#: magnitude below Alpaca's finest accepted increment (1e-9 shares).
QTY_EPS = 1e-10


def as_qty(value) -> int | float:
    """Coerce a broker/DB quantity, preserving a fraction if there is one.

    Returns an `int` when the value is whole (so `repr`, `==` against ints, and
    integer DB columns behave exactly as before) and a `float` otherwise.
    """
    f = float(value)
    i = int(f)
    return i if f == i else f


def is_zero(qty: float) -> bool:
    """True when `qty` is zero to within float noise."""
    return abs(qty) < QTY_EPS


def qty_eq(a: float, b: float) -> bool:
    """Equality for quantities. `a == b` is unsafe once quantities are floats:
    a full exit is detected as `delta == -held`, and `0.1 + 0.2 != 0.3` would
    turn a full liquidation into a partial one that leaves dust behind."""
    return abs(a - b) < QTY_EPS


def truncate_qty(qty: float, precision: int) -> float:
    """Truncate (never round up) to `precision` decimal places.

    Rounding UP would buy more than the cash-floor check sized and could
    overdraw the account by a few cents per name; rounding down cannot.
    """
    if precision < 0:
        raise ValueError(f"precision must be >= 0; got {precision}")
    scale = 10 ** precision
    return int(qty * scale) / scale


def fmt_qty(qty: float, *, width: int = 0) -> str:
    """Render a quantity for logs. Whole numbers print as integers (so existing
    log lines are unchanged); fractions print without trailing-zero noise."""
    s = (
        str(int(qty)) if float(qty) == int(qty)
        else f"{qty:.9f}".rstrip("0").rstrip(".")
    )
    return s.rjust(width) if width else s
