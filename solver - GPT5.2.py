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

# Result status
SOLVED = "SOLVED"
TIMEOUT = "TIMEOUT"
UNSOLVABLE = "UNSOLVABLE"
ERROR = "ERROR"


def _hints_from_decided_line(mask, n):
    """Compute hints from a fully decided line bitmask."""
    out = []
    run = 0
    for i in range(n):
        if mask & (1 << i):
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
def _consistent_masks(n, hints, must1, must0):
    """Return tuple of all pattern masks consistent with must1/must0."""
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


def _prewarm_patterns(n, hints_list):
    for hints in hints_list:
        _patterns_for_hints(n, tuple(hints))


class NonogramSolver:
    """Nonogram solver with bitmask line filtering + iterative DFS stack.

    State representation:
    - row_filled/row_empty: per-row bitmask
    - col_filled/col_empty: per-col bitmask
    - trail: list of prior masks for undo
    - stack: explicit DFS stack frames (r, c, choices, choice_index, trail_marker)
    """

    def __init__(self, col_hints, row_hints):
        self.row_hints = [tuple(h) for h in row_hints]
        self.col_hints = [tuple(h) for h in col_hints]
        self.height = len(self.row_hints)
        self.width = len(self.col_hints)

        self.row_fullmask = (1 << self.width) - 1
        self.col_fullmask = (1 << self.height) - 1

        self.row_filled = [0] * self.height
        self.row_empty = [0] * self.height
        self.col_filled = [0] * self.width
        self.col_empty = [0] * self.width

    def set_state(self, state):
        self.row_filled = list(state["row_filled"])
        self.row_empty = list(state["row_empty"])
        self.col_filled = list(state["col_filled"])
        self.col_empty = list(state["col_empty"])

    def get_state(self, trail, stack, need_new_guess, initialized):
        return {
            "row_filled": list(self.row_filled),
            "row_empty": list(self.row_empty),
            "col_filled": list(self.col_filled),
            "col_empty": list(self.col_empty),
            "trail": list(trail),
            "stack": list(stack),
            "need_new_guess": need_new_guess,
            "initialized": initialized,
        }

    def is_solved(self):
        for r in range(self.height):
            if (self.row_filled[r] | self.row_empty[r]) != self.row_fullmask:
                return False
        return True

    def set_cell(self, r, c, v, trail):
        mask = 1 << c
        rmask = 1 << r
        row_filled = self.row_filled[r]
        row_empty = self.row_empty[r]
        col_filled = self.col_filled[c]
        col_empty = self.col_empty[c]

        if v == FILLED:
            if row_empty & mask:
                return False
            if row_filled & mask:
                return True
            trail.append((r, c, row_filled, row_empty, col_filled, col_empty))
            self.row_filled[r] = row_filled | mask
            self.col_filled[c] = col_filled | rmask
            return True

        if row_filled & mask:
            return False
        if row_empty & mask:
            return True
        trail.append((r, c, row_filled, row_empty, col_filled, col_empty))
        self.row_empty[r] = row_empty | mask
        self.col_empty[c] = col_empty | rmask
        return True

    def undo_to(self, trail, marker):
        while len(trail) > marker:
            r, c, row_filled, row_empty, col_filled, col_empty = trail.pop()
            self.row_filled[r] = row_filled
            self.row_empty[r] = row_empty
            self.col_filled[c] = col_filled
            self.col_empty[c] = col_empty

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
                must1 = self.row_filled[idx]
                must0 = self.row_empty[idx]
                if (must1 & must0) != 0:
                    return False
                masks = _consistent_masks(self.width, self.row_hints[idx], must1, must0)
                if not masks:
                    return False

                common1 = self.row_fullmask
                common0 = self.row_fullmask
                for m in masks:
                    common1 &= m
                    common0 &= (~m) & self.row_fullmask

                unknown = ~(must1 | must0) & self.row_fullmask
                new_filled = common1 & unknown
                new_empty = common0 & unknown

                while new_filled:
                    bit = new_filled & -new_filled
                    c = bit.bit_length() - 1
                    if not self.set_cell(idx, c, FILLED, trail):
                        return False
                    if not in_col[c]:
                        q.append((False, c))
                        in_col[c] = True
                    new_filled &= new_filled - 1

                while new_empty:
                    bit = new_empty & -new_empty
                    c = bit.bit_length() - 1
                    if not self.set_cell(idx, c, EMPTY, trail):
                        return False
                    if not in_col[c]:
                        q.append((False, c))
                        in_col[c] = True
                    new_empty &= new_empty - 1
            else:
                in_col[idx] = False
                must1 = self.col_filled[idx]
                must0 = self.col_empty[idx]
                if (must1 & must0) != 0:
                    return False
                masks = _consistent_masks(self.height, self.col_hints[idx], must1, must0)
                if not masks:
                    return False

                common1 = self.col_fullmask
                common0 = self.col_fullmask
                for m in masks:
                    common1 &= m
                    common0 &= (~m) & self.col_fullmask

                unknown = ~(must1 | must0) & self.col_fullmask
                new_filled = common1 & unknown
                new_empty = common0 & unknown

                while new_filled:
                    bit = new_filled & -new_filled
                    r = bit.bit_length() - 1
                    if not self.set_cell(r, idx, FILLED, trail):
                        return False
                    if not in_row[r]:
                        q.append((True, r))
                        in_row[r] = True
                    new_filled &= new_filled - 1

                while new_empty:
                    bit = new_empty & -new_empty
                    r = bit.bit_length() - 1
                    if not self.set_cell(r, idx, EMPTY, trail):
                        return False
                    if not in_row[r]:
                        q.append((True, r))
                        in_row[r] = True
                    new_empty &= new_empty - 1

        return True

    def _choose_guess(self):
        """MRV: pick the row/col with fewest consistent masks, then a good split cell."""
        best = None

        for r in range(self.height):
            must1 = self.row_filled[r]
            must0 = self.row_empty[r]
            if (must1 | must0) == self.row_fullmask:
                continue
            masks = _consistent_masks(self.width, self.row_hints[r], must1, must0)
            if not masks:
                return None
            if best is None or len(masks) < best[0]:
                best = (len(masks), True, r, masks, must1, must0)

        for c in range(self.width):
            must1 = self.col_filled[c]
            must0 = self.col_empty[c]
            if (must1 | must0) == self.col_fullmask:
                continue
            masks = _consistent_masks(self.height, self.col_hints[c], must1, must0)
            if not masks:
                return None
            if best is None or len(masks) < best[0]:
                best = (len(masks), False, c, masks, must1, must0)

        if best is None:
            return None

        _, is_row, idx, masks, must1, must0 = best
        total = len(masks)

        best_pos = None
        best_balance = None
        best_pref = None

        n = self.width if is_row else self.height
        unknown = ~(must1 | must0) & ((1 << n) - 1)
        pos = 0
        while unknown:
            bit = unknown & -unknown
            pos = bit.bit_length() - 1
            cnt1 = 0
            for m in masks:
                if m & bit:
                    cnt1 += 1
            if cnt1 != 0 and cnt1 != total:
                balance = abs(2 * cnt1 - total)
                if best_pos is None or balance < best_balance:
                    best_pos = pos
                    best_balance = balance
                    best_pref = FILLED if cnt1 * 2 >= total else EMPTY
            unknown &= unknown - 1

        if best_pos is None:
            for pos in range(n):
                bit = 1 << pos
                if (must1 | must0) & bit == 0:
                    best_pos = pos
                    best_pref = FILLED
                    break

        if is_row:
            r, c = idx, best_pos
        else:
            r, c = best_pos, idx

        return (r, c, best_pref)

    def to_grid(self):
        grid = [[0] * self.width for _ in range(self.height)]
        for r in range(self.height):
            row_mask = self.row_filled[r]
            for c in range(self.width):
                if row_mask & (1 << c):
                    grid[r][c] = FILLED
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


def _solve_timeslice(payload):
    """Worker entry: solve one puzzle for a time slice and return status + checkpoint."""
    pid = payload["pid"]
    col_hints = payload["col_hints"]
    row_hints = payload["row_hints"]
    slice_seconds = payload["slice_seconds"]
    state = payload["state"]

    _prewarm_patterns(25, row_hints)
    _prewarm_patterns(25, col_hints)

    solver = NonogramSolver(col_hints, row_hints)

    if state:
        # Restore checkpoint: grid masks + DFS stack + undo trail.
        solver.set_state(state)
        trail = list(state["trail"])
        stack = list(state["stack"])
        need_new_guess = state["need_new_guess"]
        initialized = state["initialized"]
    else:
        # New puzzle starts with empty masks and empty DFS stack.
        trail = []
        stack = []
        need_new_guess = True
        initialized = False

    deadline = time.perf_counter() + slice_seconds
    node_count = 0
    check_interval = 256

    if not initialized:
        # Initial full propagation from empty grid (records in trail).
        if not solver.propagate(trail, None):
            return {"pid": pid, "status": UNSOLVABLE, "state": None, "solution": None}
        initialized = True

    while True:
        node_count += 1
        if node_count % check_interval == 0 and time.perf_counter() >= deadline:
            checkpoint = solver.get_state(trail, stack, need_new_guess, initialized)
            return {"pid": pid, "status": TIMEOUT, "state": checkpoint, "solution": None}

        if solver.is_solved():
            grid = solver.to_grid()
            if verify_solution(grid, col_hints, row_hints):
                return {"pid": pid, "status": SOLVED, "state": None, "solution": grid}
            return {"pid": pid, "status": ERROR, "state": None, "solution": None}

        if need_new_guess:
            choice = solver._choose_guess()
            if choice is None:
                if not stack:
                    return {"pid": pid, "status": UNSOLVABLE, "state": None, "solution": None}
                stack[-1][3] += 1
                need_new_guess = False
                continue
            r, c, pref = choice
            other = EMPTY if pref == FILLED else FILLED
            stack.append([r, c, [pref, other], 0, len(trail)])
            need_new_guess = False

        if not stack:
            return {"pid": pid, "status": UNSOLVABLE, "state": None, "solution": None}

        r, c, choices, idx, marker = stack[-1]
        if idx >= len(choices):
            stack.pop()
            solver.undo_to(trail, marker)
            if not stack:
                return {"pid": pid, "status": UNSOLVABLE, "state": None, "solution": None}
            stack[-1][3] += 1
            need_new_guess = False
            continue

        solver.undo_to(trail, marker)
        if time.perf_counter() >= deadline:
            checkpoint = solver.get_state(trail, stack, need_new_guess, initialized)
            return {"pid": pid, "status": TIMEOUT, "state": checkpoint, "solution": None}

        v = choices[idx]
        if not solver.set_cell(r, c, v, trail):
            stack[-1][3] += 1
            continue

        q = deque()
        in_row = [False] * solver.height
        in_col = [False] * solver.width
        q.append((True, r))
        in_row[r] = True
        q.append((False, c))
        in_col[c] = True

        if not solver.propagate(trail, (q, in_row, in_col)):
            solver.undo_to(trail, marker)
            stack[-1][3] += 1
            continue

        need_new_guess = True


def parse_line(line):
    return list(map(int, line.replace(".", " ").strip().split()))


def read_puzzles(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    puzzles = []
    idx = 0
    while idx < len(lines):
        if lines[idx].startswith("$"):
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

            puzzles.append((pid, col_hints, row_hints))
        else:
            idx += 1
    return puzzles


def write_solution(f_out, pid, grid):
    f_out.write(f"{pid}\n")
    for row in grid:
        f_out.write("  ".join("1" if cell == FILLED else "0" for cell in row) + "\n")
    f_out.flush()


def write_unsolved(f_out, pid):
    f_out.write(f"{pid}\n")
    f_out.write("Unsolvable or Timeout\n")
    f_out.flush()


def save_checkpoint(pid, state, checkpoint_dir):
    """Checkpoint format: pickle of state dict for a puzzle id."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{pid}.pkl")
    with open(path, "wb") as f:
        pickle.dump(state, f)
    return path


def load_checkpoint(pid, checkpoint_dir):
    path = os.path.join(checkpoint_dir, f"{pid}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def solve_all(
    puzzles,
    output_file,
    max_workers,
    slice_seconds,
    max_rounds,
    checkpoint_dir,
):
    """Parallel workflow:
    1) Submit every puzzle with a fixed time slice to the process pool.
    2) TIMEOUT -> save checkpoint + enqueue for the next round.
    3) SOLVED -> verify and immediately append to result.txt.
    4) After max_rounds, mark remaining puzzles as unsolved.
    """
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write("")

    pending = [
        {"pid": pid, "col_hints": col_hints, "row_hints": row_hints, "state": None, "rounds": 0}
        for pid, col_hints, row_hints in puzzles
    ]

    solved = set()
    unsolved = set()

    with ProcessPoolExecutor(max_workers=max_workers) as executor, open(
        output_file, "a", encoding="utf-8"
    ) as f_out:
        for round_idx in range(1, max_rounds + 1):
            if not pending:
                break

            futures = {}
            for item in pending:
                payload = {
                    "pid": item["pid"],
                    "col_hints": item["col_hints"],
                    "row_hints": item["row_hints"],
                    "slice_seconds": slice_seconds,
                    "state": item["state"],
                }
                futures[executor.submit(_solve_timeslice, payload)] = item

            next_pending = []
            for fut in as_completed(futures):
                item = futures[fut]
                pid = item["pid"]
                try:
                    result = fut.result()
                except Exception:
                    result = {"pid": pid, "status": ERROR, "state": None, "solution": None}

                status = result["status"]
                if status == SOLVED:
                    write_solution(f_out, pid, result["solution"])
                    solved.add(pid)
                elif status == TIMEOUT:
                    checkpoint = result["state"]
                    save_checkpoint(pid, checkpoint, checkpoint_dir)
                    item["state"] = checkpoint
                    item["rounds"] += 1
                    next_pending.append(item)
                elif status == UNSOLVABLE:
                    write_unsolved(f_out, pid)
                    unsolved.add(pid)
                else:
                    item["state"] = result.get("state")
                    item["rounds"] += 1
                    next_pending.append(item)

            pending = next_pending

        for item in pending:
            pid = item["pid"]
            if pid in solved or pid in unsolved:
                continue
            write_unsolved(f_out, pid)


def main():
    # Self-test examples:
    # 1) Single puzzle: python solver - GPT5.2.py --single
    # 2) Full run: python solver - GPT5.2.py
    # 3) Checkpoint recovery: python solver - GPT5.2.py --slice-seconds 0.01 --max-rounds 1
    #    Then re-run with larger slice to finish.
    parser = argparse.ArgumentParser(description="Nonogram solver with checkpoints and multiprocessing.")
    parser.add_argument("--input", default="taai2019.txt", help="input file path")
    parser.add_argument("--output", default="result.txt", help="output file path (fixed to result.txt)")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count(), help="max worker processes")
    parser.add_argument("--slice-seconds", type=float, default=30.0, help="seconds per time slice")
    parser.add_argument("--max-rounds", type=int, default=10, help="max rounds for deferred puzzles")
    parser.add_argument("--single", action="store_true", help="solve only the first puzzle")
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="checkpoint directory (default: checkpoints)"
    )
    args = parser.parse_args()

    if args.output != "result.txt":
        args.output = "result.txt"

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    puzzles = read_puzzles(args.input)
    if args.single and puzzles:
        puzzles = [puzzles[0]]

    solve_all(
        puzzles=puzzles,
        output_file=args.output,
        max_workers=args.max_workers,
        slice_seconds=args.slice_seconds,
        max_rounds=args.max_rounds,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
