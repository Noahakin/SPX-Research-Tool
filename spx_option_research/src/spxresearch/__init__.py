"""Reusable SPX option-strategy research components."""

from .data_loader import SPXSurfaceArchive
from .execution import ExecutionModel
from .option_selector import select_expiration, select_put_by_delta, select_put_spread

__all__ = [
    "ExecutionModel",
    "SPXSurfaceArchive",
    "select_expiration",
    "select_put_by_delta",
    "select_put_spread",
]

