"""Pruning method package."""

from .base import BasePruningMethod
from .hsic_lasso import HsicLassoPruner

__all__ = ["BasePruningMethod", "HsicLassoPruner"]
