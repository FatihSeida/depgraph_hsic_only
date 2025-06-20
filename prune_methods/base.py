"""Base pruning method definitions."""

from __future__ import annotations

import abc


class BasePruningMethod(abc.ABC):
    """Abstract base class for pruning methods."""

    @abc.abstractmethod
    def prune(self, model):
        """Prune the provided model.

        Parameters
        ----------
        model : Any
            Model to prune.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_config(self, cfg_path):
        """Load configuration from ``cfg_path``.

        Parameters
        ----------
        cfg_path : str or PathLike
            Path to configuration file.
        """
        raise NotImplementedError
