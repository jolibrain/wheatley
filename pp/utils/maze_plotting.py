"""
Backward-compatible import for maze plotting utilities.
"""
from pp.domains.maze.maze_plotting import *  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith("_")]
