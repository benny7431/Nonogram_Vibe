import argparse
import os
import pickle
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

# Constants
UNKNOWN = -1
EMPTY = 0
FILLED = 1


@lru_cache(maxsize=None)
def _patterns_for_hints(n, hints):
    """Return tuple of filled-cell bitmasks for all placements of hints on length n."""
    hints = tuple(hints)
    if not hints:
        return (0,)

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
        rem_blocks = m - (i + 1)
        rem_min = suffix_sum[i + 1] + (rem_blocks if rem_blocks > 0 else 0)
        latest_start = n - (block_len + rem_min)
        if latest_start < pos:
            return

        block_bits = (1 << block_len) - 1
        for start in range(pos, latest_start + 1):
            new_mask = mask | (block_bits << start)
            next_pos = start + block_len
            if i + 1 < m:
                next_pos += 1
            rec(next_pos, i + 1, new_mask)

    rec(0, 0, 0)
    return tuple(res)


@lru_cache(maxsize=200000)
def _consistent_masks_cached(n, hints, must1, must0):
    """LRU cached consistent masks for (n, hints, must1, must0)."""
    fullmask = (1 << n) - 1
    patterns = _patterns_for_hints(n, hints)
    out = []
    for mask in patterns:
        if (mask & must0) != 0:
            continue
        if (mask & must1) != must1:
            continue
        out.append(mask)
    return tuple(out)


def _hints_from_mask(n, mask):
    """Compute hints from a fully decided mask (bit=1 means filled)."""
    out = []
    run = 0
    for i in range(n):
        if mask & (1 << i):
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


class NonogramSolver:
    """
    Iterative Nonogram solver with explicit stack (checkpointable).

    State structure (all pickleable):
    - row_must1/row_must0: list[int], bitmasks per row
    - col_must1/col_must0: list[int], bitmasks per col
    - trail: list of (r, c, prev_row1, prev_row0, prev_col1, prev_col0)
    - stack: list of decision frames: {r, c, choices, next_idx, trail_len}
    """

    def __init__(self, col_hints, row_hints):
        self.row_hints = [tuple(h) for h in row_hints]
        self.col_hints = [tuple(h) for h in col_hints]
        self.height = len(self.row_hints)
        self.width = len(self.col_hints)

        self.row_fullmask = (1 << self.width) - 1
        self.col_fullmask = (1 << self.height) - 1

        self.row_must1 = [0] * self.height
        self.row_must0 = [0] * self.height
        self.col_must1 = [0] * self.width
        self.col_must0 = [0] * self.width

        self.trail = []
        self.stack = []
        self.initialized = False

    def export_state(self):
        return {
            "row_must1": self.row_must1,
            "row_must0": self.row_must0,
            "col_must1": self.col_must1,
            "col_must0": self.col_must0,
            "trail": self.trail,
            "stack": self.stack,
        }

    def load_state(self, state):
        self.row_must1 = state["row_must1"]
        self.row_must0 = state["row_must0"]
        self.col_must1 = state["col_must1"]
        self.col_must0 = state["col_must0"]
        self.trail = state["trail"]
        self.stack = state["stack"]
        self.initialized = True

    def is_solved(self):
        for r in range(self.height):
            if (self.row_must1[r] | self.row_must0[r]) != self.row_fullmask:
                return False
        return True

    def _set_cell(self, r, c, v, trail):
        bitc = 1 << c
        bitr = 1 << r
        row1 = self.row_must1[r]
        row0 = self.row_must0[r]
        col1 = self.col_must1[c]
        col0 = self.col_must0[c]

        if v == FILLED:
            if (row0 & bitc) or (col0 & bitr):
                return False
            if (row1 & bitc) and (col1 & bitr):
                return True
            trail.append((r, c, row1, row0, col1, col0))
            self.row_must1[r] = row1 | bitc
            self.col_must1[c] = col1 | bitr
            return True

        if (row1 & bitc) or (col1 & bitr):
            return False
        if (row0 & bitc) and (col0 & bitr):
            return True
        trail.append((r, c, row1, row0, col1, col0))
        self.row_must0[r] = row0 | bitc
        self.col_must0[c] = col0 | bitr
        return True

    def _undo_to(self, trail_len):
        while len(self.trail) > trail_len:
            r, c, row1, row0, col1, col0 = self.trail.pop()
            self.row_must1[r] = row1
            self.row_must0[r] = row0
            self.col_must1[c] = col1
            self.col_must0[c] = col0

    def _solve_line_masks(self, n, hints, must1, must0, fullmask):
        if must1 & must0:
            return None

        if (must1 | must0) == fullmask:
            if _hints_from_mask(n, must1) != list(hints):
                return None
            return must1, must0

        masks = _consistent_masks_cached(n, hints, must1, must0)
        if not masks:
            return None

        common1 = fullmask
        common0 = fullmask
        for m in masks:
            common1 &= m
            common0 &= (~m) & fullmask

        new_must1 = must1 | common1
        new_must0 = must0 | common0
        if new_must1 & new_must0:
            return None
        return new_must1, new_must0

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
                must1 = self.row_must1[idx]
                must0 = self.row_must0[idx]
                new_masks = self._solve_line_masks(
                    self.width, self.row_hints[idx], must1, must0, self.row_fullmask
                )
                if new_masks is None:
                    return False
                new_must1, new_must0 = new_masks

                delta1 = new_must1 & ~must1
                delta0 = new_must0 & ~must0
                while delta1:
                    bit = delta1 & -delta1
                    c = bit.bit_length() - 1
                    if not self._set_cell(idx, c, FILLED, trail):
                        return False
                    if not in_col[c]:
                        q.append((False, c))
                        in_col[c] = True
                    delta1 &= delta1 - 1
                while delta0:
                    bit = delta0 & -delta0
                    c = bit.bit_length() - 1
                    if not self._set_cell(idx, c, EMPTY, trail):
                        return False
                    if not in_col[c]:
                        q.append((False, c))
                        in_col[c] = True
                    delta0 &= delta0 - 1
            else:
                in_col[idx] = False
                must1 = self.col_must1[idx]
                must0 = self.col_must0[idx]
                new_masks = self._solve_line_masks(
                    self.height, self.col_hints[idx], must1, must0, self.col_fullmask
                )
                if new_masks is None:
                    return False
                new_must1, new_must0 = new_masks

                delta1 = new_must1 & ~must1
                delta0 = new_must0 & ~must0
                while delta1:
                    bit = delta1 & -delta1
                    r = bit.bit_length() - 1
                    if not self._set_cell(r, idx, FILLED, trail):
                        return False
                    if not in_row[r]:
                        q.append((True, r))
                        in_row[r] = True
                    delta1 &= delta1 - 1
                while delta0:
                    bit = delta0 & -delta0
                    r = bit.bit_length() - 1
                    if not self._set_cell(r, idx, EMPTY, trail):
                        return False
                    if not in_row[r]:
                        q.append((True, r))
                        in_row[r] = True
                    delta0 &= delta0 - 1

        return True

    def _choose_guess(self):
        """MRV: pick the row/col with fewest consistent masks, then a good split cell."""
        best = None

        for r in range(self.height):
            must1 = self.row_must1[r]
            must0 = self.row_must0[r]
            if (must1 | must0) == self.row_fullmask:
                continue
            masks = _consistent_masks_cached(self.width, self.row_hints[r], must1, must0)
            if not masks:
                return None
            if best is None or len(masks) < best[0]:
                best = (len(masks), True, r, masks, must1, must0)

        for c in range(self.width):
            must1 = self.col_must1[c]
            must0 = self.col_must0[c]
            if (must1 | must0) == self.col_fullmask:
                continue
            masks = _consistent_masks_cached(self.height, self.col_hints[c], must1, must0)
            if not masks:
                return None
            if best is None or len(masks) < best[0]:
                best = (len(masks), False, c, masks, must1, must0)

        if best is None:
            return None

        _, is_row, idx, masks, must1, must0 = best
        total = len(masks)
        n = self.width if is_row else self.height
        fullmask = (1 << n) - 1
        unknown_mask = (~(must1 | must0)) & fullmask

        best_pos = None
        best_balance = None
        best_pref = None

        temp = unknown_mask
        while temp:
            bit = temp & -temp
            pos = bit.bit_length() - 1
            cnt1 = 0
            for m in masks:
                if m & bit:
                    cnt1 += 1
            if 0 < cnt1 < total:
                balance = abs(2 * cnt1 - total)
                if best_pos is None or balance < best_balance:
                    best_pos = pos
                    best_balance = balance
                    best_pref = FILLED if cnt1 * 2 >= total else EMPTY
            temp &= temp - 1

        if best_pos is None:
            best_pos = (unknown_mask & -unknown_mask).bit_length() - 1
            best_pref = FILLED

        if is_row:
            r, c = idx, best_pos
        else:
            r, c = best_pos, idx

        return r, c, best_pref

    def _try_choice(self, r, c, v):
        if not self._set_cell(r, c, v, self.trail):
            return False

        q = deque()
        in_row = [False] * self.height
        in_col = [False] * self.width
        q.append((True, r))
        in_row[r] = True
        q.append((False, c))
        in_col[c] = True
        return self.propagate(self.trail, (q, in_row, in_col))

    def _apply_frame_choice(self, frame):
        while frame["next_idx"] < len(frame["choices"]):
            v = frame["choices"][frame["next_idx"]]
            frame["next_idx"] += 1
            self._undo_to(frame["trail_len"])
            if self._try_choice(frame["r"], frame["c"], v):
                return True
        self._undo_to(frame["trail_len"])
        return False

    def _backtrack(self):
        while self.stack:
            frame = self.stack[-1]
            if self._apply_frame_choice(frame):
                return True
            self.stack.pop()
        return False

    def solve_timeslice(self, deadline, check_interval=512):
        """Solve until deadline. Returns (status, payload)."""
        if not self.initialized:
            if not self.propagate(self.trail, queue=None):
                return "UNSOLVED", None
            self.initialized = True
            if self.is_solved():
                return "SOLVED", self.grid_as_list()

        nodes = 0
        while True:
            if (nodes & (check_interval - 1)) == 0:
                if time.perf_counter() >= deadline:
                    return "TIMEOUT", self.export_state()

            if self.is_solved():
                return "SOLVED", self.grid_as_list()

            choice = self._choose_guess()
            if choice is None:
                if not self._backtrack():
                    return "UNSOLVED", None
                nodes += 1
                continue

            r, c, pref = choice
            frame = {
                "r": r,
                "c": c,
                "choices": [pref, EMPTY if pref == FILLED else FILLED],
                "next_idx": 0,
                "trail_len": len(self.trail),
            }
            self.stack.append(frame)
            if not self._apply_frame_choice(frame):
                if not self._backtrack():
                    return "UNSOLVED", None
            nodes += 1

    def grid_as_list(self):
        grid = []
        for r in range(self.height):
            row = []
            mask = self.row_must1[r]
            for c in range(self.width):
                row.append(1 if (mask & (1 << c)) else 0)
            grid.append(row)
        return grid


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


def read_puzzles(input_file, limit=None):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    puzzles = []
    idx = 0
    while idx < len(lines):
        if lines[idx].startswith('$'):
            pid = lines[idx]
            idx += 1

            col_hints = []
            for _ in range(25):
                col_hints.append(parse_line(lines[idx]))
                idx += 1

            row_hints = []
            for _ in range(25):
                row_hints.append(parse_line(lines[idx]))
                idx += 1

            puzzles.append({"pid": pid, "col_hints": col_hints, "row_hints": row_hints})
            if limit and len(puzzles) >= limit:
                break
        else:
            idx += 1
    return puzzles


def prewarm_patterns(puzzles):
    """Precompute patterns for all hints to warm caches."""
    hints_set = set()
    for puzzle in puzzles:
        for hints in puzzle["col_hints"] + puzzle["row_hints"]:
            hints_set.add(tuple(hints))

    for hints in hints_set:
        _patterns_for_hints(25, hints)


def solve_timeslice_worker(pid, col_hints, row_hints, state, slice_seconds):
    solver = NonogramSolver(col_hints, row_hints)
    if state is not None:
        solver.load_state(state)

    t0 = time.perf_counter()
    deadline = time.perf_counter() + slice_seconds
    status, payload = solver.solve_timeslice(deadline)
    elapsed = time.perf_counter() - t0

    if status == "SOLVED":
        if verify_solution(payload, col_hints, row_hints):
            return pid, "SOLVED", payload, elapsed
        return pid, "UNSOLVED", None, elapsed

    if status == "TIMEOUT":
        return pid, "TIMEOUT", payload, elapsed

    return pid, "UNSOLVED", None, elapsed


def write_solution(f_out, pid, grid):
    f_out.write(f"{pid}\n")
    for row in grid:
        f_out.write("  ".join("1" if cell == FILLED else "0" for cell in row) + "\n")
    f_out.flush()


def write_unsolved(f_out, pid):
    f_out.write(f"{pid}\n")
    f_out.write("Unsolvable or Timeout\n")
    f_out.flush()


def ensure_checkpoint_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def checkpoint_path(checkpoint_dir, pid):
    safe_pid = pid.replace('$', '$')
    return os.path.join(checkpoint_dir, f"{safe_pid}.pkl")


def main():
    parser = argparse.ArgumentParser(description="Nonogram solver with time slicing + checkpoints")
    parser.add_argument("--input-file", default="taai2019.txt")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    parser.add_argument("--slice-seconds", type=float, default=30)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--single", action="store_true", help="Solve only the first puzzle")
    args = parser.parse_args()

    output_file = "result.txt"
    checkpoint_dir = "checkpoints"
    ensure_checkpoint_dir(checkpoint_dir)

    puzzles = read_puzzles(args.input_file, limit=1 if args.single else None)
    if not puzzles:
        print("No puzzles found.")
        return

    prewarm_patterns(puzzles)

    solved = set()
    deferred = {}
    total_times = {}

    with open(output_file, "w", encoding="utf-8") as f_out:
        for round_idx in range(1, args.max_rounds + 1):
            if round_idx == 1:
                current = []
                for puzzle in puzzles:
                    pid = puzzle["pid"]
                    state = None
                    path = checkpoint_path(checkpoint_dir, pid)
                    if os.path.exists(path):
                        with open(path, "rb") as f_in:
                            state = pickle.load(f_in)
                    current.append((pid, puzzle["col_hints"], puzzle["row_hints"], state))
            else:
                current = []
                for pid, state in deferred.items():
                    puzzle = next(p for p in puzzles if p["pid"] == pid)
                    current.append((pid, puzzle["col_hints"], puzzle["row_hints"], state))

            if not current:
                break

            deferred = {}
            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [
                    executor.submit(
                        solve_timeslice_worker, pid, col_hints, row_hints, state, args.slice_seconds
                    )
                    for pid, col_hints, row_hints, state in current
                ]

                for future in as_completed(futures):
                    pid, status, payload, elapsed = future.result()
                    total_times[pid] = total_times.get(pid, 0.0) + elapsed
                    if status == "SOLVED":
                        write_solution(f_out, pid, payload)
                        print(f"{pid} solved in {total_times[pid]:.4f}s")
                        solved.add(pid)
                        path = checkpoint_path(checkpoint_dir, pid)
                        if os.path.exists(path):
                            os.remove(path)
                    elif status == "TIMEOUT":
                        deferred[pid] = payload
                        path = checkpoint_path(checkpoint_dir, pid)
                        with open(path, "wb") as f_cp:
                            pickle.dump(payload, f_cp)
                    else:
                        write_unsolved(f_out, pid)
                        print(f"{pid} unsolved in {total_times[pid]:.4f}s")
                        solved.add(pid)

        for pid, state in deferred.items():
            if pid in solved:
                continue
            write_unsolved(f_out, pid)
            print(f"{pid} unsolved in {total_times.get(pid, 0.0):.4f}s")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()

"""
Self-test examples (no external packages required):
1) Single puzzle: python "solver - GPT5.2.py" --single
2) Full run: python "solver - GPT5.2.py"
3) Checkpoint resume: python "solver - GPT5.2.py" --slice-seconds 0.01
   then rerun with a larger slice to resume.
"""
