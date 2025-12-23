import sys
import time
from collections import deque
from functools import lru_cache

# Constants
UNKNOWN = -1
EMPTY = 0
FILLED = 1


def _hints_from_decided_line(line):
    """Compute hints from a fully decided line containing only 0/1."""
    out = []
    run = 0
    for v in line:
        if v == FILLED:
            run += 1
        else:
            if run:
                out.append(run)
                run = 0
    if run:
        out.append(run)
    return out


@lru_cache(maxsize=None)
def _patterns_for_hints(n, hints):
    """Return tuple of filled-cell bitmasks for all placements of hints on length n."""
    hints = tuple(hints)
    if not hints:
        return (0,)

    # Precompute suffix sums for tight bounds.
    m = len(hints)
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + hints[i]

    res = []

    def rec(pos, i, mask):
        if i == m:
            res.append(mask)
            return

        block_len = hints[i]
        # Remaining blocks after i.
        rem_blocks = m - (i + 1)
        rem_min = suffix_sum[i + 1] + (rem_blocks if rem_blocks > 0 else 0)
        latest_start = n - (block_len + rem_min)
        if latest_start < pos:
            return

        block_bits = (1 << block_len) - 1
        for start in range(pos, latest_start + 1):
            new_mask = mask | (block_bits << start)
            next_pos = start + block_len
            # Mandatory separator if there are more blocks.
            if i + 1 < m:
                next_pos += 1
            rec(next_pos, i + 1, new_mask)

    rec(0, 0, 0)
    return tuple(res)


class NonogramSolver:
    """Competition-style Nonogram solver: fast line filtering + MRV backtracking."""

    def __init__(self, col_hints, row_hints):
        self.row_hints = [tuple(h) for h in row_hints]
        self.col_hints = [tuple(h) for h in col_hints]
        self.height = len(self.row_hints)
        self.width = len(self.col_hints)

        self.row_fullmask = (1 << self.width) - 1
        self.col_fullmask = (1 << self.height) - 1

        # Grid: -1=Unknown, 0=Empty, 1=Filled
        self.grid = [[UNKNOWN] * self.width for _ in range(self.height)]

        # Cache of consistent masks per (n, hints, must1, must0)
        self._consistent_cache = {}

    def solve(self):
        # Top-level propagate once. Keep changes on success.
        trail = []
        if not self.propagate(trail, queue=None):
            self.undo(trail)
            return None

        if self._dfs():
            return self.grid

        # No solution found.
        self.undo(trail)
        return None

    def is_solved(self):
        for r in range(self.height):
            if UNKNOWN in self.grid[r]:
                return False
        return True

    def set_cell(self, r, c, v, trail):
        cur = self.grid[r][c]
        if cur == v:
            return True
        if cur != UNKNOWN:
            return False
        trail.append((r, c, cur))
        self.grid[r][c] = v
        return True

    def undo(self, trail):
        while trail:
            r, c, old = trail.pop()
            self.grid[r][c] = old

    def _line_to_constraints(self, line):
        must1 = 0
        must0 = 0
        for i, v in enumerate(line):
            bit = 1 << i
            if v == FILLED:
                must1 |= bit
            elif v == EMPTY:
                must0 |= bit
        return must1, must0

    def _consistent_masks(self, n, hints, must1, must0):
        key = (n, hints, must1, must0)
        cached = self._consistent_cache.get(key)
        if cached is not None:
            return cached

        fullmask = (1 << n) - 1
        patterns = _patterns_for_hints(n, hints)
        out = []
        for mask in patterns:
            # filled cells cannot overlap must0; must1 must be subset of filled.
            if (mask & must0) != 0:
                continue
            if (mask & must1) != must1:
                continue
            out.append(mask)

        out = tuple(out)
        self._consistent_cache[key] = out
        return out

    def solve_line(self, line, hints, n, fullmask):
        """Return updated line list, or None on contradiction."""
        # Fast contradiction on fully decided line.
        if UNKNOWN not in line:
            if tuple(_hints_from_decided_line(line)) != hints:
                return None
            return list(line)

        must1, must0 = self._line_to_constraints(line)
        masks = self._consistent_masks(n, hints, must1, must0)
        if not masks:
            return None

        common1 = fullmask
        common0 = fullmask
        for m in masks:
            common1 &= m
            common0 &= (~m) & fullmask

        out = list(line)
        for i in range(n):
            bit = 1 << i
            if common1 & bit:
                out[i] = FILLED
            elif common0 & bit:
                out[i] = EMPTY
        return out

    def propagate(self, trail, queue):
        """AC-style propagation on affected rows/cols. Records changes in trail."""
        if queue is None:
            q = deque()
            in_row = [False] * self.height
            in_col = [False] * self.width
            for r in range(self.height):
                q.append((True, r))
                in_row[r] = True
            for c in range(self.width):
                q.append((False, c))
                in_col[c] = True
        else:
            q, in_row, in_col = queue

        while q:
            is_row, idx = q.popleft()
            if is_row:
                in_row[idx] = False
                line = self.grid[idx]
                hints = self.row_hints[idx]
                n = self.width
                fullmask = self.row_fullmask
                new_line = self.solve_line(line, hints, n, fullmask)
                if new_line is None:
                    return False

                for c, (old, new) in enumerate(zip(line, new_line)):
                    if old != new:
                        if not self.set_cell(idx, c, new, trail):
                            return False
                        if not in_col[c]:
                            q.append((False, c))
                            in_col[c] = True
            else:
                in_col[idx] = False
                line = [self.grid[r][idx] for r in range(self.height)]
                hints = self.col_hints[idx]
                n = self.height
                fullmask = self.col_fullmask
                new_line = self.solve_line(line, hints, n, fullmask)
                if new_line is None:
                    return False

                for r, (old, new) in enumerate(zip(line, new_line)):
                    if old != new:
                        if not self.set_cell(r, idx, new, trail):
                            return False
                        if not in_row[r]:
                            q.append((True, r))
                            in_row[r] = True

        return True

    def _choose_guess(self):
        """MRV: pick the row/col with fewest consistent masks, then a good split cell."""
        best = None

        # Rows
        for r in range(self.height):
            line = self.grid[r]
            if UNKNOWN not in line:
                continue
            must1, must0 = self._line_to_constraints(line)
            masks = self._consistent_masks(self.width, self.row_hints[r], must1, must0)
            if not masks:
                return None
            if best is None or len(masks) < best[0]:
                best = (len(masks), True, r, masks, line)

        # Cols
        for c in range(self.width):
            line = [self.grid[r][c] for r in range(self.height)]
            if UNKNOWN not in line:
                continue
            must1, must0 = self._line_to_constraints(line)
            masks = self._consistent_masks(self.height, self.col_hints[c], must1, must0)
            if not masks:
                return None
            if best is None or len(masks) < best[0]:
                best = (len(masks), False, c, masks, line)

        if best is None:
            return None

        _, is_row, idx, masks, line = best
        total = len(masks)

        # Pick a cell that splits masks near half.
        best_pos = None
        best_balance = None
        best_pref = None

        n = self.width if is_row else self.height
        for pos in range(n):
            if line[pos] != UNKNOWN:
                continue
            bit = 1 << pos
            cnt1 = 0
            for m in masks:
                if m & bit:
                    cnt1 += 1
            if cnt1 == 0 or cnt1 == total:
                continue
            balance = abs(2 * cnt1 - total)
            if best_pos is None or balance < best_balance:
                best_pos = pos
                best_balance = balance
                best_pref = FILLED if cnt1 * 2 >= total else EMPTY

        # Fallback if all unknown positions are forced in one direction (rare).
        if best_pos is None:
            for pos in range(n):
                if line[pos] == UNKNOWN:
                    best_pos = pos
                    best_pref = FILLED
                    break

        if is_row:
            r, c = idx, best_pos
        else:
            r, c = best_pos, idx

        return (r, c, best_pref)

    def _dfs(self):
        if self.is_solved():
            return True

        choice = self._choose_guess()
        if choice is None:
            return False
        r, c, pref = choice

        for v in (pref, EMPTY if pref == FILLED else FILLED):
            local_trail = []
            if not self.set_cell(r, c, v, local_trail):
                self.undo(local_trail)
                continue

            # Start propagation only from affected row/col.
            q = deque()
            in_row = [False] * self.height
            in_col = [False] * self.width
            q.append((True, r))
            in_row[r] = True
            q.append((False, c))
            in_col[c] = True

            if self.propagate(local_trail, (q, in_row, in_col)):
                if self._dfs():
                    return True

            self.undo(local_trail)

        return False


def verify_solution(grid, col_hints, row_hints):
    """Check if a solved grid matches hints."""
    if not grid:
        return False

    rows = len(grid)
    cols = len(grid[0])

    def get_hints(line):
        h = []
        count = 0
        for x in line:
            if x == 1:
                count += 1
            elif count > 0:
                h.append(count)
                count = 0
        if count > 0:
            h.append(count)
        return h

    for r in range(rows):
        if get_hints(grid[r]) != row_hints[r]:
            return False

    for c in range(cols):
        col_line = [grid[r][c] for r in range(rows)]
        if get_hints(col_line) != col_hints[c]:
            return False

    return True


def parse_line(line):
    return list(map(int, line.replace('.', ' ').strip().split()))


def main():
    input_file = 'taai2019.txt'
    output_file = 'result_py.txt'

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    with open(output_file, 'w', encoding='utf-8') as f_out:
        idx = 0
        while idx < len(lines):
            if lines[idx].startswith('$'):
                pid = lines[idx]
                idx += 1

                col_hints = []
                for _ in range(25):
                    if idx < len(lines):
                        col_hints.append(parse_line(lines[idx]))
                        idx += 1

                row_hints = []
                for _ in range(25):
                    if idx < len(lines):
                        row_hints.append(parse_line(lines[idx]))
                        idx += 1

                print(f"Solving {pid}...", end=" ", flush=True)
                t0 = time.time()

                solver = NonogramSolver(col_hints, row_hints)
                sol = solver.solve()

                t1 = time.time()

                is_correct = False
                if sol:
                    is_correct = verify_solution(sol, col_hints, row_hints)

                print(f"Done. Time: {t1 - t0:.4f}s. Correct: {is_correct}")

                f_out.write(f"{pid}\n")
                if sol and is_correct:
                    for row in sol:
                        f_out.write("  ".join("1" if cell == FILLED else "0" for cell in row) + "\n")
                else:
                    f_out.write("Unsolvable or Error\n")
            else:
                idx += 1


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
