"""Fractional repair windows shared by the engine and examples."""

from decimal import Decimal


def repair_ratio(value: str | float) -> float:
    ratio = float(value)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("repair window must be a finite fraction in [0, 1]")
    return ratio


def repair_token_counts(length: int, begin: float, end: float) -> tuple[int, int]:
    """Floor fractions of the original length and count overlap only once."""
    left = int(length * Decimal(str(repair_ratio(begin))))
    right = min(int(length * Decimal(str(repair_ratio(end)))), length - left)
    return left, right
