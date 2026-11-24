"""設定值檢查：錯誤訊息寫出設定類別與欄位名稱，方便找到要改的參數。"""

import math
from typing import Callable


def _check(config: object, names: tuple[str, ...], valid: Callable[[object], bool], requirement: str) -> None:
    owner = type(config).__name__
    for name in names:
        value = getattr(config, name)
        if not valid(value):
            raise ValueError(f"{owner}.{name} must be {requirement} (got {value!r})")


def require_finite(config: object, *names: str) -> None:
    _check(config, names, math.isfinite, "finite")


def require_positive(config: object, *names: str) -> None:
    _check(config, names, lambda value: math.isfinite(value) and value > 0.0, "positive and finite")


def require_nonnegative(config: object, *names: str) -> None:
    _check(config, names, lambda value: math.isfinite(value) and value >= 0.0,
           "non-negative and finite")


def require_finite_values(config: object, *names: str) -> None:
    """欄位是數值序列，例如 candidate_offsets。"""
    _check(config, names, lambda values: all(math.isfinite(value) for value in values),
           "a sequence of finite values")


def require_nonnegative_values(config: object, *names: str) -> None:
    _check(config, names, lambda values: all(math.isfinite(value) and value >= 0.0 for value in values),
           "a sequence of non-negative finite values")
