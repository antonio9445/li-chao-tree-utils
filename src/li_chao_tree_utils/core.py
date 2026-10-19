"""Li Chao Tree for maximum queries over linear functions.

A Li Chao Tree stores a family of lines y = m*x + b and, for a fixed
coordinate range [lo, hi], answers "which line gives the maximum value at
x?" in O(log R) time per operation, where R = hi - lo is the width of the
range. The tree is built over discrete integer coordinates; the function
evaluated at x is the line that wins at that exact point.

Design decisions (stated plainly so the tests and README agree):

1. We compare floats using a configurable ``eps`` (default 1e-9). Floating
   point arithmetic makes strict ``==`` unreliable for deciding whether
   two lines tie at a point. The tree's correctness does not depend on the
   choice of tiebreaker; any consistent one works. We just must not crash
   or return nonsense. Using ``eps`` keeps the internal comparisons
   stable across platforms.

2. The coordinate range is a closed interval ``[lo, hi]`` of integers. The
   tree subdivides by integer midpoints. Query points must lie in this
   range. The range is fixed at construction. If you need a different
   range, build a new tree.

3. Adding the same line object twice has no extra effect beyond inserting
   another copy into the structure. We do not deduplicate; the caller
   controls identity.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


class Line:
    """A linear function y = m*x + b.

    Instances are plain value objects: two ``Line`` objects are equal when
    their slopes and intercepts are close within ``eps``. Equality is used
    only for the caller's convenience; the tree itself never relies on it.
    """

    __slots__ = ("m", "b")

    def __init__(self, m: float, b: float) -> None:
        self.m = float(m)
        self.b = float(b)

    def eval(self, x: float) -> float:
        """Return the value of this line at ``x``."""
        return self.m * x + self.b

    def __repr__(self) -> str:
        return f"Line(m={self.m!r}, b={self.b!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Line):
            return NotImplemented
        return (
            abs(self.m - other.m) <= 1e-9
            and abs(self.b - other.b) <= 1e-9
        )

    def __hash__(self) -> int:
        # Round for hashing consistency with __eq__.
        return hash((round(self.m, 9), round(self.b, 9)))


def _fmax(a: float, b: float, eps: float) -> Tuple[float, int]:
    """Return (value, winner) where winner is 0 for ``a``, 1 for ``b``.

    When the two values are within ``eps`` of each other, we break the tie
    deterministically by preferring the first argument. This keeps the
    recursion terminating: without a stable tiebreaker, swaps could ping-
    pong between levels. The direction of the tiebreak does not affect
    correctness of the final answer.
    """
    if a >= b - eps:
        return a, 0
    return b, 1


class LiChaoTree:
    """A Li Chao Tree over the integer range ``[lo, hi]``.

    Maintains a dynamic set of lines and answers maximum-value queries.
    Each query returns the :class:`Line` currently stored in the tree that
    attains the maximum value at the query point. If multiple lines tie,
    one of them is returned; the choice is deterministic but unspecified.

    Parameters
    ----------
    lo, hi:
        Inclusive integer bounds of the query coordinate range.
    eps:
        Absolute tolerance used when comparing float values. Defaults to
        ``1e-9``. Increase it if your slopes/intercepts are large and you
        want ties to be recognised more generously.

    Notes
    -----
    The internal array is sized as ``4 * (hi - lo + 1)`` to safely cover
    the segment-tree recursion without index bounds checks on the hot path.
    This trades memory for simplicity and speed.
    """

    def __init__(self, lo: int, hi: int, eps: float = 1e-9) -> None:
        if lo > hi:
            raise ValueError(
                f"lo must be <= hi; got lo={lo}, hi={hi}"
            )
        if lo != int(lo) or hi != int(hi):
            raise ValueError("lo and hi must be integers")
        lo = int(lo)
        hi = int(hi)

        self.lo: int = lo
        self.hi: int = hi
        self.eps: float = float(eps)
        self.size: int = hi - lo + 1
        # 4x the range is plenty for a segment tree on this interval.
        self._tree: List[Optional[Line]] = [None] * (4 * max(self.size, 1))

    # -- internal helpers ------------------------------------------------

    def _node_lo_hi(self, node: int, seg_lo: int, seg_hi: int) -> int:
        """Return the midpoint of ``[seg_lo, seg_hi]``.

        Factored out so the insertion logic reads as a straight-line
        divide-and-conquer.
        """
        return (seg_lo + seg_hi) // 2

    def _insert(
        self,
        node: int,
        seg_lo: int,
        seg_hi: int,
        new_line: Line,
    ) -> None:
        """Insert ``new_line`` into the subtree rooted at ``node``.

        The classic Li Chao invariant: at each node we keep the line that
        wins at the midpoint, and push the loser down to the side where it
        could still win.
        """
        cur = self._tree[node]
        if cur is None:
            self._tree[node] = new_line
            return

        mid = self._node_lo_hi(node, seg_lo, seg_hi)

        cur_at_mid = cur.eval(mid)
        new_at_mid = new_line.eval(mid)

        best_at_mid, winner = _fmax(cur_at_mid, new_at_mid, self.eps)
        if winner == 0:
            keeper, loser = cur, new_line
        else:
            keeper, loser = new_line, cur

        self._tree[node] = keeper

        # Push the loser into the half where it might still beat the keeper.
        # At seg_lo, compare; at seg_hi, compare. The loser can only win on
        # the side where it beats the keeper at the endpoint.
        if seg_lo == seg_hi:
            return

        loser_at_lo = loser.eval(seg_lo)
        keeper_at_lo = keeper.eval(seg_lo)
        if loser_at_lo > keeper_at_lo + self.eps:
            self._insert(2 * node + 1, seg_lo, mid, loser)

        loser_at_hi = loser.eval(seg_hi)
        keeper_at_hi = keeper.eval(seg_hi)
        if loser_at_hi > keeper_at_hi + self.eps:
            self._insert(2 * node + 2, mid + 1, seg_hi, loser)

    def _query(
        self,
        node: int,
        seg_lo: int,
        seg_hi: int,
        x: int,
    ) -> Tuple[float, Optional[Line]]:
        """Return (value, line) for the best line stored at or below
        ``node`` for coordinate ``x``.

        Returns ``(float('-inf'), None)`` if no line covers this point.
        """
        cur = self._tree[node]
        best_val = float("-inf")
        best_line: Optional[Line] = None

        if cur is not None:
            best_val = cur.eval(x)
            best_line = cur

        if seg_lo == seg_hi:
            return best_val, best_line

        mid = self._node_lo_hi(node, seg_lo, seg_hi)
        if x <= mid:
            child_val, child_line = self._query(
                2 * node + 1, seg_lo, mid, x
            )
        else:
            child_val, child_line = self._query(
                2 * node + 2, mid + 1, seg_hi, x
            )

        if child_line is not None and child_val > best_val + self.eps:
            return child_val, child_line
        return best_val, best_line

    # -- public API ------------------------------------------------------

    def add_line(self, m: float, b: float) -> Line:
        """Insert the line y = m*x + b into the tree.

        Returns the inserted :class:`Line` for the caller's convenience
        (e.g. for tracking or display). The tree does not deduplicate:
        adding the same line twice is allowed.
        """
        line = Line(m, b)
        self._insert(0, self.lo, self.hi, line)
        return line

    def query_max(self, x: int) -> Optional[Line]:
        """Return the line attaining the maximum value at ``x``.

        Returns ``None`` if no line has been added. The argument ``x`` must
        be an integer in ``[lo, hi]``; otherwise a :class:`ValueError` is
        raised. Non-integer coordinates are not supported because the tree
        subdivides at integer midpoints.
        """
        if x != int(x):
            raise ValueError(
                f"query point must be an integer; got {x!r}"
            )
        x = int(x)
        if x < self.lo or x > self.hi:
            raise ValueError(
                f"query point {x} out of range [{self.lo}, {self.hi}]"
            )
        _, line = self._query(0, self.lo, self.hi, x)
        return line

    def query_max_value(self, x: int) -> float:
        """Return the maximum value attained at ``x``.

        Returns ``float('-inf')`` if no line has been added.
        """
        if x != int(x):
            raise ValueError(
                f"query point must be an integer; got {x!r}"
            )
        x = int(x)
        if x < self.lo or x > self.hi:
            raise ValueError(
                f"query point {x} out of range [{self.lo}, {self.hi}]"
            )
        val, _ = self._query(0, self.lo, self.hi, x)
        return val
