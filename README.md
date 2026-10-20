# li-chao-tree-utils

A Li Chao Tree that maintains a set of linear functions and answers, for any integer point `x` in a fixed range, which function attains the maximum value there. Insertion and query are both O(log R) where R is the range width.

## Usage

```python
from li_chao_tree_utils import LiChaoTree, Line

# Range is a closed integer interval [lo, hi].
tree = LiChaoTree(lo=0, hi=100)

tree.add_line(2.0, 0.0)   # y = 2x
tree.add_line(-1.0, 50.0) # y = -x + 50

best = tree.query_max(25)        # returns the winning Line
best_value = tree.query_max_value(25)  # returns the float value

print(best.m, best.b, best_value)  # -1.0 50.0 25.0
```

## Why this exists

The problem is simple to state: you have a changing family of lines and you repeatedly need the maximum at specific points. A segment tree over the coordinate range, where each node keeps the line winning at its midpoint and pushes the loser toward the side where it could still win, gives O(log R) per operation without any coordinate compression or dynamic allocation. The trade-off is that the range is fixed at construction and the tree uses O(R) memory, so it suits bounded-coordinate work rather than sparse huge ranges.

## The awkward edge

The tree subdivides at integer midpoints, so query points must be integers in `[lo, hi]`. Floats are compared with a tolerance (`eps`, default `1e-9`); when two lines tie exactly at a point, the tree returns one of them deterministically, but which one is an implementation detail. If you need a specific tiebreaker, compare the returned value yourself. Non-integer query points raise `ValueError`, as do out-of-range queries.

## Exported names

- `LiChaoTree(lo, hi, eps=1e-9)` — the tree; methods `add_line(m, b) -> Line`, `query_max(x) -> Line | None`, `query_max_value(x) -> float`.
- `Line(m, b)` — value object with `.m`, `.b`, and `.eval(x) -> float`.
