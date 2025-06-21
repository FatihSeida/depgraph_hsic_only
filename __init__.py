"""depgraph_hsic_only package entry point."""

from .prune_methods import HsicLassoPruner
from .main import main

__all__ = ["HsicLassoPruner", "main"]

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
