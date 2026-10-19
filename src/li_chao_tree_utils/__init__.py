"""li-chao-tree-utils: A Li Chao Tree for maximum queries over linear functions.

Maintains a set of linear functions and answers "which function is largest
at x?" in O(log C) per operation, where C is the fixed coordinate range.
See README for the intended use case and the single design decision that
matters: floats are compared with tolerance, never with ==.
"""

from .core import LiChaoTree, Line

__all__ = ["LiChaoTree", "Line"]
