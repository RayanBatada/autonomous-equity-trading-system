"""Train / val / test window date constants.

Three windows because two isn't enough. Train/val lets you overfit to val.
Train/val/test keeps test as a one-shot honesty check that the auto-research
agent in Phase 6 will never see.

Changing TEST_START or TEST end is a code change requiring code review,
same posture as the risk rails. The harness gates test access in eval.py.
"""

from datetime import date
from typing import Literal

TRAIN_START = date(2023, 1, 1)
TRAIN_END = date(2025, 6, 30)

VAL_START = date(2025, 7, 1)
VAL_END = date(2025, 12, 31)

TEST_START = date(2026, 1, 1)
# TEST_END is "today" at evaluation time; computed at call.

WindowName = Literal["train", "val", "test"]


def window_dates(name: WindowName) -> tuple[date, date]:
    if name == "train":
        return (TRAIN_START, TRAIN_END)
    if name == "val":
        return (VAL_START, VAL_END)
    if name == "test":
        return (TEST_START, date.today())
    raise ValueError(f"Unknown window: {name!r}; expected 'train', 'val', or 'test'.")
