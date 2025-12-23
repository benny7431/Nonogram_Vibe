
import sys
import time
from collections import deque

# Constants
UNKNOWN = -1
EMPTY = 0
FILLED = 1


class NonogramSolver:
    """
    25x25 競賽取向加速版：
    - propagate: deque + O(1) 去重 flags
    - line solver: 位元遮罩枚舉所有 block placements，O(#placements) 且用 bit ops 快速檢查衝突
    - backtrack: 用 (row_options * col_options) 最小的 UNKNOWN cell 做猜測
    """

    def __init__(self, col_hints, row_hints):
        self.height = 25
        self.width = 25
        self.col_hints = [tuple(h) for h in col_hints]
        self.row_hints = [tuple(h) for h in row_hints]
        self.grid = [[UNKNOWN] * self.width for _ in range(self.height)]

        # line solver cache: (line_tuple, hints_tuple) -> (forced_line_tuple, options_count, filled_or, filled_and)
        self._line_cache = {}

        # bitmask for width/height=25
        self._full_mask_w = (1 << self.width) - 1
        self._full_mask_h = (1 << self.height) - 1

    def solve(self):
        if not self.propagate():
            return None
        if self.is_solved():
            return self.grid
        return self.backtrack()

    def is_solved(self):
        for r in range(self.height):
            if UNKNOWN in self.grid[r]:
                return False
        return True

    # -----------------------------
    # Propagation (fast queue)
    # -----------------------------
    def propagate(self):
        q = deque()
        inq_row = [True] * self.height
        inq_col = [True] * self.width

        for r in range(self.height):
            q.append((True, r))
        for c in range(self.width):
            q.append((False, c))

        while q:
            is_row, idx = q.popleft()
            if is_row:
                inq_row[idx] = False
                line = tuple(self.grid[idx])
                hints = self.row_hints[idx]
                forced, _, _, _ = self.solve_line_cached(line, hints)
                if forced is None:
                    return False

                # apply diffs
                if forced != line:
                    for c in range(self.width):
                        nv = forced[c]
                        if nv != line[c]:
                            self.grid[idx][c] = nv
                            if not inq_col[c]:
                                inq_col[c] = True
                                q.append((False, c))
            else:
                inq_col[idx] = False
                col = tuple(self.grid[r][idx] for r in range(self.height))
                hints = self.col_hints[idx]
                forced, _, _, _ = self.solve_line_cached(col, hints)
                if forced is None:
                    return False

                if forced != col:
                    for r in range(self.height):
                        nv = forced[r]
                        if nv != col[r]:
                            self.grid[r][idx] = nv
                            if not inq_row[r]:
                                inq_row[r] = True
                                q.append((True, r))
        return True

    # -----------------------------
    # Line solver (bitmask placements)
    # -----------------------------
    def solve_line_cached(self, line, hints):
        key = (line, hints)
        hit = self._line_cache.get(key)
        if hit is not None:
            return hit
        res = self._solve_line_bitmask(line, hints)
        self._line_cache[key] = res
        return res

    def _solve_line_bitmask(self, line, hints):
        """
        回傳 (forced_line_tuple, options_count, filled_or, filled_and)
        forced_line: 只推出必然為 FILLED / EMPTY 的格，其餘維持原值
        """
        n = len(line)
        full = (1 << n) - 1

        known_f = 0
        known_e = 0
        for i, v in enumerate(line):
            if v == FILLED:
                known_f |= (1 << i)
            elif v == EMPTY:
                known_e |= (1 << i)

        # quick: empty hints
        if not hints:
            # all empty; contradiction if any known filled
            if known_f:
                return (None, 0, 0, 0)
            forced = list(line)
            for i in range(n):
                forced[i] = EMPTY
            return (tuple(forced), 1, 0, 0)

        k = len(hints)

        # precompute suffix minimal length from block i
        suf_min = [0] * (k + 1)
        # suf_min[i] = sum(hints[i:]) + (k-1-i)   (spaces between remaining blocks)
        s = 0
        for i in range(k - 1, -1, -1):
            s += hints[i]
            suf_min[i] = s + (k - 1 - i)
        suf_min[k] = 0

        options = 0
        filled_and = full
        filled_or = 0

        # small helper for mask range [a,b] inclusive
        def mask_range(a, b):
            if a > b:
                return 0
            return ((1 << (b - a + 1)) - 1) << a

        sys.setrecursionlimit(10000)

        def rec(block_i, pos, filled_mask):
            nonlocal options, filled_and, filled_or

            # prune: any known FILLED before pos must already be covered
            if pos > 0:
                pref = (1 << pos) - 1
                if (known_f & pref & ~filled_mask) != 0:
                    return

            if block_i == k:
                # remaining tail is empty; must not contain known filled
                tail_mask = mask_range(pos, n - 1)
                if (known_f & tail_mask) != 0:
                    return
                # overall conflicts
                if (known_e & filled_mask) != 0:
                    return
                if (known_f & ~filled_mask) != 0:
                    return

                options += 1
                filled_and &= filled_mask
                filled_or |= filled_mask
                return

            bl = hints[block_i]

            # remaining minimal length (including this block and following spaces)
            rem = suf_min[block_i]
            latest_start = n - rem
            # start s in [pos, latest_start]
            for s in range(pos, latest_start + 1):
                # cells pos..s-1 are empty
                if s > pos:
                    skipped = mask_range(pos, s - 1)
                    if (known_f & skipped) != 0:
                        continue

                block_mask = ((1 << bl) - 1) << s
                if (known_e & block_mask) != 0:
                    continue

                next_pos = s + bl

                # separator if not last block
                if block_i < k - 1:
                    if next_pos >= n:
                        continue
                    sep_bit = 1 << next_pos
                    # separator must be empty
                    if (known_f & sep_bit) != 0:
                        continue
                    # line may already force FILLED there
                    if line[next_pos] == FILLED:
                        continue
                    next_pos += 1

                rec(block_i + 1, next_pos, filled_mask | block_mask)

        rec(0, 0, 0)

        if options == 0:
            return (None, 0, 0, 0)

        forced = list(line)
        for i in range(n):
            bit = 1 << i
            if (filled_and & bit) != 0:
                forced[i] = FILLED
            elif (filled_or & bit) == 0:
                forced[i] = EMPTY
            # else keep original (UNKNOWN or fixed)

        return (tuple(forced), options, filled_or, filled_and)

    # -----------------------------
    # Backtracking (better branching)
    # -----------------------------
    def backtrack(self):
        # choose best unknown cell: minimize row_options * col_options
        best = None  # (score, r, c)
        row_stats = [None] * self.height
        col_stats = [None] * self.width

        # compute row stats only when needed
        for r in range(self.height):
            if UNKNOWN not in self.grid[r]:
                continue
            line = tuple(self.grid[r])
            forced, opts, row_or, row_and = self.solve_line_cached(line, self.row_hints[r])
            if forced is None:
                return None
            row_stats[r] = (opts, row_or, row_and)

        for c in range(self.width):
            col = tuple(self.grid[r][c] for r in range(self.height))
            if UNKNOWN not in col:
                continue
            forced, opts, col_or, col_and = self.solve_line_cached(col, self.col_hints[c])
            if forced is None:
                return None
            col_stats[c] = (opts, col_or, col_and)

        for r in range(self.height):
            if row_stats[r] is None:
                continue
            ropts, ror, rand = row_stats[r]
            for c in range(self.width):
                if self.grid[r][c] != UNKNOWN:
                    continue
                if col_stats[c] is None:
                    continue
                copts, cor, cand = col_stats[c]

                score = ropts * copts

                # tie-break: prefer cells that are closer to forced (one side says always filled or never filled)
                bit_r = 1 << c
                bit_c = 1 << r
                tight = 0
                if (rand & bit_r) != 0 or (ror & bit_r) == 0:
                    tight += 1
                if (cand & bit_c) != 0 or (cor & bit_c) == 0:
                    tight += 1

                key = (score, -tight, r, c)
                if best is None or key < best[0]:
                    best = (key, r, c)

        if best is None:
            return self.grid  # solved

        _, r, c = best

        saved_grid = [row[:] for row in self.grid]

        # decide try order using line stats
        ropts, ror, rand = row_stats[r]
        copts, cor, cand = col_stats[c]
        can_fill = ((ror >> c) & 1) == 1 and ((cor >> r) & 1) == 1
        must_fill = ((rand >> c) & 1) == 1 or ((cand >> r) & 1) == 1
        can_empty = not must_fill  # because if always-filled then cannot be empty

        try_values = []
        if must_fill:
            try_values = [FILLED]
        else:
            # heuristic: try the value that looks more constrained first
            # if either side says "never filled", fill is impossible
            if not can_fill:
                try_values = [EMPTY]
            else:
                # bias toward FILLED tends to prune faster on many puzzle sets
                try_values = [FILLED, EMPTY]

        for v in try_values:
            self.grid[r][c] = v
            if self.propagate():
                res = self.backtrack()
                if res is not None:
                    return res
            self.grid = [row[:] for row in saved_grid]

        return None


def verify_solution(grid, col_hints, row_hints):
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
                start_time = time.time()

                solver = NonogramSolver(col_hints, row_hints)
                solution = solver.solve()

                end_time = time.time()

                is_correct = False
                if solution:
                    is_correct = verify_solution(solution, col_hints, row_hints)

                print(f"Done. Time: {end_time - start_time:.4f}s. Correct: {is_correct}")

                f_out.write(f"{pid}\n")
                if solution and is_correct:
                    for row in solution:
                        f_out.write("  ".join("1" if x == FILLED else "0" for x in row) + "\n")
                else:
                    f_out.write("Unsolvable or Error\n")
            else:
                idx += 1


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
