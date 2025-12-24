#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kUnknown = -1;
constexpr int kEmpty = 0;
constexpr int kFilled = 1;

struct HintsKey {
  int n = 0;
  std::vector<int> hints;

  bool operator==(const HintsKey& other) const {
    return n == other.n && hints == other.hints;
  }
};

struct HintsKeyHash {
  std::size_t operator()(const HintsKey& key) const {
    std::size_t h = std::hash<int>{}(key.n);
    for (int v : key.hints) {
      h ^= std::hash<int>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
  }
};

struct ConsistentKey {
  int n = 0;
  std::vector<int> hints;
  uint32_t must1 = 0;
  uint32_t must0 = 0;

  bool operator==(const ConsistentKey& other) const {
    return n == other.n && must1 == other.must1 && must0 == other.must0 &&
           hints == other.hints;
  }
};

struct ConsistentKeyHash {
  std::size_t operator()(const ConsistentKey& key) const {
    std::size_t h = std::hash<int>{}(key.n);
    for (int v : key.hints) {
      h ^= std::hash<int>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    h ^= std::hash<uint32_t>{}(key.must1) + 0x9e3779b97f4a7c15ULL + (h << 6) +
         (h >> 2);
    h ^= std::hash<uint32_t>{}(key.must0) + 0x9e3779b97f4a7c15ULL + (h << 6) +
         (h >> 2);
    return h;
  }
};

std::unordered_map<HintsKey, std::vector<uint32_t>, HintsKeyHash> g_patterns_cache;
std::unordered_map<ConsistentKey, std::vector<uint32_t>, ConsistentKeyHash>
    g_consistent_cache;
std::mutex g_cache_mutex;

std::vector<uint32_t> PatternsForHints(int n, const std::vector<int>& hints) {
  HintsKey key{n, hints};
  {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    auto it = g_patterns_cache.find(key);
    if (it != g_patterns_cache.end()) {
      return it->second;
    }
  }

  if (hints.empty()) {
    return {0U};
  }

  const int m = static_cast<int>(hints.size());
  std::vector<int> suffix_sum(m + 1, 0);
  for (int i = m - 1; i >= 0; --i) {
    suffix_sum[i] = suffix_sum[i + 1] + hints[i];
  }

  std::vector<uint32_t> res;
  std::function<void(int, int, uint32_t)> rec = [&](int pos, int i, uint32_t mask) {
    if (i == m) {
      res.push_back(mask);
      return;
    }
    int block_len = hints[i];
    int rem_blocks = m - (i + 1);
    int rem_min = suffix_sum[i + 1] + (rem_blocks > 0 ? rem_blocks : 0);
    int latest_start = n - (block_len + rem_min);
    if (latest_start < pos) {
      return;
    }
    uint32_t block_bits = (block_len >= 32) ? 0xFFFFFFFFU : ((1U << block_len) - 1U);
    for (int start = pos; start <= latest_start; ++start) {
      uint32_t new_mask = mask | (block_bits << start);
      int next_pos = start + block_len;
      if (i + 1 < m) {
        next_pos += 1;
      }
      rec(next_pos, i + 1, new_mask);
    }
  };

  rec(0, 0, 0U);
  {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_patterns_cache.emplace(key, res);
  }
  return res;
}

std::vector<uint32_t> ConsistentMasksCached(int n,
                                            const std::vector<int>& hints,
                                            uint32_t must1,
                                            uint32_t must0) {
  ConsistentKey key{n, hints, must1, must0};
  {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    auto it = g_consistent_cache.find(key);
    if (it != g_consistent_cache.end()) {
      return it->second;
    }
  }

  uint32_t fullmask = (n >= 32) ? 0xFFFFFFFFU : ((1U << n) - 1U);
  std::vector<uint32_t> patterns = PatternsForHints(n, hints);
  std::vector<uint32_t> out;
  out.reserve(patterns.size());
  for (uint32_t mask : patterns) {
    if ((mask & must0) != 0) {
      continue;
    }
    if ((mask & must1) != must1) {
      continue;
    }
    out.push_back(mask);
  }

  {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_consistent_cache.emplace(key, out);
  }
  return out;
}

std::vector<int> HintsFromMask(int n, uint32_t mask) {
  std::vector<int> out;
  int run = 0;
  for (int i = 0; i < n; ++i) {
    if (mask & (1U << i)) {
      run += 1;
    } else if (run) {
      out.push_back(run);
      run = 0;
    }
  }
  if (run) {
    out.push_back(run);
  }
  return out;
}

struct TrailEntry {
  int r = 0;
  int c = 0;
  uint32_t row1 = 0;
  uint32_t row0 = 0;
  uint32_t col1 = 0;
  uint32_t col0 = 0;
};

struct Frame {
  int r = 0;
  int c = 0;
  std::vector<int> choices;
  std::size_t next_idx = 0;
  std::size_t trail_len = 0;
};

struct SolverState {
  std::vector<uint32_t> row_must1;
  std::vector<uint32_t> row_must0;
  std::vector<uint32_t> col_must1;
  std::vector<uint32_t> col_must0;
  std::vector<TrailEntry> trail;
  std::vector<Frame> stack;
  bool initialized = false;
};

class NonogramSolver {
 public:
  NonogramSolver(std::vector<std::vector<int>> col_hints,
                 std::vector<std::vector<int>> row_hints)
      : row_hints_(std::move(row_hints)),
        col_hints_(std::move(col_hints)),
        height_(static_cast<int>(row_hints_.size())),
        width_(static_cast<int>(col_hints_.size())),
        row_fullmask_((width_ >= 32) ? 0xFFFFFFFFU : ((1U << width_) - 1U)),
        col_fullmask_((height_ >= 32) ? 0xFFFFFFFFU : ((1U << height_) - 1U)),
        row_must1_(height_, 0U),
        row_must0_(height_, 0U),
        col_must1_(width_, 0U),
        col_must0_(width_, 0U) {}

  SolverState ExportState() const {
    return SolverState{row_must1_, row_must0_, col_must1_, col_must0_, trail_, stack_,
                       initialized_};
  }

  void LoadState(const SolverState& state) {
    row_must1_ = state.row_must1;
    row_must0_ = state.row_must0;
    col_must1_ = state.col_must1;
    col_must0_ = state.col_must0;
    trail_ = state.trail;
    stack_ = state.stack;
    initialized_ = state.initialized;
  }

  bool IsSolved() const {
    for (int r = 0; r < height_; ++r) {
      if ((row_must1_[r] | row_must0_[r]) != row_fullmask_) {
        return false;
      }
    }
    return true;
  }

  std::vector<std::vector<int>> GridAsList() const {
    std::vector<std::vector<int>> grid(height_, std::vector<int>(width_, 0));
    for (int r = 0; r < height_; ++r) {
      uint32_t mask = row_must1_[r];
      for (int c = 0; c < width_; ++c) {
        grid[r][c] = (mask & (1U << c)) ? 1 : 0;
      }
    }
    return grid;
  }

  std::pair<std::string, std::optional<SolverState>> SolveTimeslice(
      const std::chrono::steady_clock::time_point& deadline,
      int check_interval = 512) {
    if (!initialized_) {
      if (!Propagate(nullptr)) {
        return {"UNSOLVED", std::nullopt};
      }
      initialized_ = true;
      if (IsSolved()) {
        return {"SOLVED", std::nullopt};
      }
    }

    int nodes = 0;
    while (true) {
      if ((nodes & (check_interval - 1)) == 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
          return {"TIMEOUT", ExportState()};
        }
      }

      if (IsSolved()) {
        return {"SOLVED", std::nullopt};
      }

      auto choice = ChooseGuess();
      if (!choice.has_value()) {
        if (!Backtrack()) {
          return {"UNSOLVED", std::nullopt};
        }
        nodes += 1;
        continue;
      }

      Frame frame;
      frame.r = choice->r;
      frame.c = choice->c;
      frame.choices = {choice->pref, choice->pref == kFilled ? kEmpty : kFilled};
      frame.next_idx = 0;
      frame.trail_len = trail_.size();
      stack_.push_back(frame);
      if (!ApplyFrameChoice(stack_.back())) {
        if (!Backtrack()) {
          return {"UNSOLVED", std::nullopt};
        }
      }
      nodes += 1;
    }
  }

  bool Propagate(std::deque<std::pair<bool, int>>* queue) {
    std::deque<std::pair<bool, int>> q;
    std::vector<bool> in_row(height_, false);
    std::vector<bool> in_col(width_, false);

    if (!queue) {
      for (int r = 0; r < height_; ++r) {
        q.emplace_back(true, r);
        in_row[r] = true;
      }
      for (int c = 0; c < width_; ++c) {
        q.emplace_back(false, c);
        in_col[c] = true;
      }
    } else {
      q = std::move(*queue);
      for (const auto& item : q) {
        if (item.first) {
          in_row[item.second] = true;
        } else {
          in_col[item.second] = true;
        }
      }
    }

    while (!q.empty()) {
      auto [is_row, idx] = q.front();
      q.pop_front();
      if (is_row) {
        in_row[idx] = false;
        uint32_t must1 = row_must1_[idx];
        uint32_t must0 = row_must0_[idx];
        auto new_masks = SolveLineMasks(width_, row_hints_[idx], must1, must0, row_fullmask_);
        if (!new_masks.has_value()) {
          return false;
        }
        auto [new_must1, new_must0] = *new_masks;
        uint32_t delta1 = new_must1 & ~must1;
        uint32_t delta0 = new_must0 & ~must0;
        while (delta1) {
          uint32_t bit = delta1 & (~delta1 + 1);
          int c = __builtin_ctz(bit);
          if (!SetCell(idx, c, kFilled)) {
            return false;
          }
          if (!in_col[c]) {
            q.emplace_back(false, c);
            in_col[c] = true;
          }
          delta1 &= (delta1 - 1);
        }
        while (delta0) {
          uint32_t bit = delta0 & (~delta0 + 1);
          int c = __builtin_ctz(bit);
          if (!SetCell(idx, c, kEmpty)) {
            return false;
          }
          if (!in_col[c]) {
            q.emplace_back(false, c);
            in_col[c] = true;
          }
          delta0 &= (delta0 - 1);
        }
      } else {
        in_col[idx] = false;
        uint32_t must1 = col_must1_[idx];
        uint32_t must0 = col_must0_[idx];
        auto new_masks = SolveLineMasks(height_, col_hints_[idx], must1, must0, col_fullmask_);
        if (!new_masks.has_value()) {
          return false;
        }
        auto [new_must1, new_must0] = *new_masks;
        uint32_t delta1 = new_must1 & ~must1;
        uint32_t delta0 = new_must0 & ~must0;
        while (delta1) {
          uint32_t bit = delta1 & (~delta1 + 1);
          int r = __builtin_ctz(bit);
          if (!SetCell(r, idx, kFilled)) {
            return false;
          }
          if (!in_row[r]) {
            q.emplace_back(true, r);
            in_row[r] = true;
          }
          delta1 &= (delta1 - 1);
        }
        while (delta0) {
          uint32_t bit = delta0 & (~delta0 + 1);
          int r = __builtin_ctz(bit);
          if (!SetCell(r, idx, kEmpty)) {
            return false;
          }
          if (!in_row[r]) {
            q.emplace_back(true, r);
            in_row[r] = true;
          }
          delta0 &= (delta0 - 1);
        }
      }
    }
    return true;
  }

 private:
  struct Guess {
    int r = 0;
    int c = 0;
    int pref = kFilled;
  };

  bool SetCell(int r, int c, int v) {
    uint32_t bitc = 1U << c;
    uint32_t bitr = 1U << r;
    uint32_t row1 = row_must1_[r];
    uint32_t row0 = row_must0_[r];
    uint32_t col1 = col_must1_[c];
    uint32_t col0 = col_must0_[c];

    if (v == kFilled) {
      if ((row0 & bitc) || (col0 & bitr)) {
        return false;
      }
      if ((row1 & bitc) && (col1 & bitr)) {
        return true;
      }
      trail_.push_back({r, c, row1, row0, col1, col0});
      row_must1_[r] = row1 | bitc;
      col_must1_[c] = col1 | bitr;
      return true;
    }

    if ((row1 & bitc) || (col1 & bitr)) {
      return false;
    }
    if ((row0 & bitc) && (col0 & bitr)) {
      return true;
    }
    trail_.push_back({r, c, row1, row0, col1, col0});
    row_must0_[r] = row0 | bitc;
    col_must0_[c] = col0 | bitr;
    return true;
  }

  void UndoTo(std::size_t trail_len) {
    while (trail_.size() > trail_len) {
      TrailEntry entry = trail_.back();
      trail_.pop_back();
      row_must1_[entry.r] = entry.row1;
      row_must0_[entry.r] = entry.row0;
      col_must1_[entry.c] = entry.col1;
      col_must0_[entry.c] = entry.col0;
    }
  }

  std::optional<std::pair<uint32_t, uint32_t>> SolveLineMasks(
      int n,
      const std::vector<int>& hints,
      uint32_t must1,
      uint32_t must0,
      uint32_t fullmask) {
    if (must1 & must0) {
      return std::nullopt;
    }

    if ((must1 | must0) == fullmask) {
      if (HintsFromMask(n, must1) != hints) {
        return std::nullopt;
      }
      return std::make_pair(must1, must0);
    }

    std::vector<uint32_t> masks = ConsistentMasksCached(n, hints, must1, must0);
    if (masks.empty()) {
      return std::nullopt;
    }

    uint32_t common1 = fullmask;
    uint32_t common0 = fullmask;
    for (uint32_t m : masks) {
      common1 &= m;
      common0 &= (~m) & fullmask;
    }

    uint32_t new_must1 = must1 | common1;
    uint32_t new_must0 = must0 | common0;
    if (new_must1 & new_must0) {
      return std::nullopt;
    }
    return std::make_pair(new_must1, new_must0);
  }

  std::optional<Guess> ChooseGuess() {
    struct Candidate {
      std::size_t count = 0;
      bool is_row = true;
      int idx = 0;
      std::vector<uint32_t> masks;
      uint32_t must1 = 0;
      uint32_t must0 = 0;
    };

    std::optional<Candidate> best;
    for (int r = 0; r < height_; ++r) {
      uint32_t must1 = row_must1_[r];
      uint32_t must0 = row_must0_[r];
      if ((must1 | must0) == row_fullmask_) {
        continue;
      }
      std::vector<uint32_t> masks = ConsistentMasksCached(width_, row_hints_[r], must1, must0);
      if (masks.empty()) {
        return std::nullopt;
      }
      if (!best.has_value() || masks.size() < best->count) {
        best = Candidate{masks.size(), true, r, std::move(masks), must1, must0};
      }
    }

    for (int c = 0; c < width_; ++c) {
      uint32_t must1 = col_must1_[c];
      uint32_t must0 = col_must0_[c];
      if ((must1 | must0) == col_fullmask_) {
        continue;
      }
      std::vector<uint32_t> masks = ConsistentMasksCached(height_, col_hints_[c], must1, must0);
      if (masks.empty()) {
        return std::nullopt;
      }
      if (!best.has_value() || masks.size() < best->count) {
        best = Candidate{masks.size(), false, c, std::move(masks), must1, must0};
      }
    }

    if (!best.has_value()) {
      return std::nullopt;
    }

    const Candidate& cand = *best;
    int n = cand.is_row ? width_ : height_;
    uint32_t fullmask = (n >= 32) ? 0xFFFFFFFFU : ((1U << n) - 1U);
    uint32_t unknown_mask = (~(cand.must1 | cand.must0)) & fullmask;

    int best_pos = -1;
    int best_balance = 0;
    int best_pref = kFilled;
    uint32_t temp = unknown_mask;
    while (temp) {
      uint32_t bit = temp & (~temp + 1);
      int pos = __builtin_ctz(bit);
      int cnt1 = 0;
      for (uint32_t m : cand.masks) {
        if (m & bit) {
          cnt1 += 1;
        }
      }
      if (cnt1 > 0 && cnt1 < static_cast<int>(cand.masks.size())) {
        int balance = std::abs(2 * cnt1 - static_cast<int>(cand.masks.size()));
        if (best_pos == -1 || balance < best_balance) {
          best_pos = pos;
          best_balance = balance;
          best_pref = (cnt1 * 2 >= static_cast<int>(cand.masks.size())) ? kFilled : kEmpty;
        }
      }
      temp &= (temp - 1);
    }

    if (best_pos == -1) {
      best_pos = __builtin_ctz(unknown_mask);
      best_pref = kFilled;
    }

    Guess guess;
    if (cand.is_row) {
      guess.r = cand.idx;
      guess.c = best_pos;
    } else {
      guess.r = best_pos;
      guess.c = cand.idx;
    }
    guess.pref = best_pref;
    return guess;
  }

  bool TryChoice(int r, int c, int v) {
    if (!SetCell(r, c, v)) {
      return false;
    }
    std::deque<std::pair<bool, int>> q;
    q.emplace_back(true, r);
    q.emplace_back(false, c);
    return Propagate(&q);
  }

  bool ApplyFrameChoice(Frame& frame) {
    while (frame.next_idx < frame.choices.size()) {
      int v = frame.choices[frame.next_idx];
      frame.next_idx += 1;
      UndoTo(frame.trail_len);
      if (TryChoice(frame.r, frame.c, v)) {
        return true;
      }
    }
    UndoTo(frame.trail_len);
    return false;
  }

  bool Backtrack() {
    while (!stack_.empty()) {
      Frame& frame = stack_.back();
      if (ApplyFrameChoice(frame)) {
        return true;
      }
      stack_.pop_back();
    }
    return false;
  }

  std::vector<std::vector<int>> row_hints_;
  std::vector<std::vector<int>> col_hints_;
  int height_ = 0;
  int width_ = 0;
  uint32_t row_fullmask_ = 0;
  uint32_t col_fullmask_ = 0;
  std::vector<uint32_t> row_must1_;
  std::vector<uint32_t> row_must0_;
  std::vector<uint32_t> col_must1_;
  std::vector<uint32_t> col_must0_;
  std::vector<TrailEntry> trail_;
  std::vector<Frame> stack_;
  bool initialized_ = false;
};

struct Puzzle {
  std::string pid;
  std::vector<std::vector<int>> col_hints;
  std::vector<std::vector<int>> row_hints;
};

std::vector<int> ParseLine(const std::string& line) {
  std::string normalized = line;
  for (char& ch : normalized) {
    if (ch == '.') {
      ch = ' ';
    }
  }
  std::vector<int> out;
  std::stringstream ss(normalized);
  std::string token;
  while (ss >> token) {
    out.push_back(std::stoi(token));
  }
  return out;
}

std::vector<Puzzle> ReadPuzzles(const std::string& input_file, int limit) {
  std::ifstream input(input_file);
  std::string line;
  std::vector<std::string> lines;
  while (std::getline(input, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }

  std::vector<Puzzle> puzzles;
  std::size_t idx = 0;
  while (idx < lines.size()) {
    if (!lines[idx].empty() && lines[idx][0] == '$') {
      Puzzle puzzle;
      puzzle.pid = lines[idx];
      idx += 1;
      for (int i = 0; i < 25 && idx < lines.size(); ++i) {
        puzzle.col_hints.push_back(ParseLine(lines[idx]));
        idx += 1;
      }
      for (int i = 0; i < 25 && idx < lines.size(); ++i) {
        puzzle.row_hints.push_back(ParseLine(lines[idx]));
        idx += 1;
      }
      puzzles.push_back(std::move(puzzle));
      if (limit > 0 && static_cast<int>(puzzles.size()) >= limit) {
        break;
      }
    } else {
      idx += 1;
    }
  }
  return puzzles;
}

void PrewarmPatterns(const std::vector<Puzzle>& puzzles) {
  std::vector<std::vector<int>> hints_set;
  for (const auto& puzzle : puzzles) {
    for (const auto& hints : puzzle.col_hints) {
      hints_set.push_back(hints);
    }
    for (const auto& hints : puzzle.row_hints) {
      hints_set.push_back(hints);
    }
  }
  std::sort(hints_set.begin(), hints_set.end());
  hints_set.erase(std::unique(hints_set.begin(), hints_set.end()), hints_set.end());
  for (const auto& hints : hints_set) {
    PatternsForHints(25, hints);
  }
}

bool VerifySolution(const std::vector<std::vector<int>>& grid,
                    const std::vector<std::vector<int>>& col_hints,
                    const std::vector<std::vector<int>>& row_hints) {
  if (grid.empty()) {
    return false;
  }
  int rows = static_cast<int>(grid.size());
  int cols = static_cast<int>(grid.front().size());
  auto get_hints = [](const std::vector<int>& line) {
    std::vector<int> hints;
    int count = 0;
    for (int cell : line) {
      if (cell == 1) {
        count += 1;
      } else if (count > 0) {
        hints.push_back(count);
        count = 0;
      }
    }
    if (count > 0) {
      hints.push_back(count);
    }
    return hints;
  };

  for (int r = 0; r < rows; ++r) {
    if (get_hints(grid[r]) != row_hints[r]) {
      return false;
    }
  }
  for (int c = 0; c < cols; ++c) {
    std::vector<int> col_line(rows);
    for (int r = 0; r < rows; ++r) {
      col_line[r] = grid[r][c];
    }
    if (get_hints(col_line) != col_hints[c]) {
      return false;
    }
  }
  return true;
}

void WriteSolution(std::ofstream& out, const std::string& pid,
                   const std::vector<std::vector<int>>& grid) {
  out << pid << "\n";
  for (const auto& row : grid) {
    for (std::size_t i = 0; i < row.size(); ++i) {
      out << (row[i] == kFilled ? "1" : "0");
      if (i + 1 < row.size()) {
        out << "  ";
      }
    }
    out << "\n";
  }
  out.flush();
}

void WriteUnsolved(std::ofstream& out, const std::string& pid) {
  out << pid << "\n";
  out << "Unsolvable or Timeout\n";
  out.flush();
}

bool EnsureCheckpointDir(const std::string& path) {
  std::ostringstream cmd;
  cmd << "mkdir -p " << path;
  return std::system(cmd.str().c_str()) == 0;
}

std::string CheckpointPath(const std::string& checkpoint_dir, const std::string& pid) {
  return checkpoint_dir + "/" + pid + ".bin";
}

void WriteState(const std::string& path, const SolverState& state) {
  std::ofstream out(path, std::ios::binary);
  auto write_vec = [&out](const auto& vec) {
    uint64_t size = vec.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size > 0) {
      out.write(reinterpret_cast<const char*>(vec.data()), sizeof(vec[0]) * vec.size());
    }
  };
  write_vec(state.row_must1);
  write_vec(state.row_must0);
  write_vec(state.col_must1);
  write_vec(state.col_must0);

  uint64_t trail_size = state.trail.size();
  out.write(reinterpret_cast<const char*>(&trail_size), sizeof(trail_size));
  for (const auto& entry : state.trail) {
    out.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
  }

  uint64_t stack_size = state.stack.size();
  out.write(reinterpret_cast<const char*>(&stack_size), sizeof(stack_size));
  for (const auto& frame : state.stack) {
    out.write(reinterpret_cast<const char*>(&frame.r), sizeof(frame.r));
    out.write(reinterpret_cast<const char*>(&frame.c), sizeof(frame.c));
    uint64_t choices_size = frame.choices.size();
    out.write(reinterpret_cast<const char*>(&choices_size), sizeof(choices_size));
    out.write(reinterpret_cast<const char*>(frame.choices.data()),
              sizeof(frame.choices[0]) * frame.choices.size());
    out.write(reinterpret_cast<const char*>(&frame.next_idx), sizeof(frame.next_idx));
    out.write(reinterpret_cast<const char*>(&frame.trail_len), sizeof(frame.trail_len));
  }

  out.write(reinterpret_cast<const char*>(&state.initialized), sizeof(state.initialized));
}

std::optional<SolverState> ReadState(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return std::nullopt;
  }
  SolverState state;
  auto read_vec = [&in](auto& vec) {
    uint64_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    vec.resize(size);
    if (size > 0) {
      in.read(reinterpret_cast<char*>(vec.data()), sizeof(vec[0]) * vec.size());
    }
  };
  read_vec(state.row_must1);
  read_vec(state.row_must0);
  read_vec(state.col_must1);
  read_vec(state.col_must0);

  uint64_t trail_size = 0;
  in.read(reinterpret_cast<char*>(&trail_size), sizeof(trail_size));
  state.trail.resize(trail_size);
  for (uint64_t i = 0; i < trail_size; ++i) {
    in.read(reinterpret_cast<char*>(&state.trail[i]), sizeof(state.trail[i]));
  }

  uint64_t stack_size = 0;
  in.read(reinterpret_cast<char*>(&stack_size), sizeof(stack_size));
  state.stack.resize(stack_size);
  for (uint64_t i = 0; i < stack_size; ++i) {
    Frame frame;
    in.read(reinterpret_cast<char*>(&frame.r), sizeof(frame.r));
    in.read(reinterpret_cast<char*>(&frame.c), sizeof(frame.c));
    uint64_t choices_size = 0;
    in.read(reinterpret_cast<char*>(&choices_size), sizeof(choices_size));
    frame.choices.resize(choices_size);
    in.read(reinterpret_cast<char*>(frame.choices.data()),
            sizeof(frame.choices[0]) * frame.choices.size());
    in.read(reinterpret_cast<char*>(&frame.next_idx), sizeof(frame.next_idx));
    in.read(reinterpret_cast<char*>(&frame.trail_len), sizeof(frame.trail_len));
    state.stack[i] = std::move(frame);
  }

  in.read(reinterpret_cast<char*>(&state.initialized), sizeof(state.initialized));
  if (!in) {
    return std::nullopt;
  }
  return state;
}

struct WorkerResult {
  std::string pid;
  std::string status;
  std::optional<SolverState> state;
  std::vector<std::vector<int>> grid;
  double elapsed = 0.0;
};

WorkerResult SolveTimesliceWorker(const std::string& pid,
                                  const std::vector<std::vector<int>>& col_hints,
                                  const std::vector<std::vector<int>>& row_hints,
                                  const std::optional<SolverState>& state,
                                  double slice_seconds) {
  NonogramSolver solver(col_hints, row_hints);
  if (state.has_value()) {
    solver.LoadState(*state);
  }

  auto t0 = std::chrono::steady_clock::now();
  auto deadline = t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                std::chrono::duration<double>(slice_seconds));
  auto [status, payload] = solver.SolveTimeslice(deadline);
  auto elapsed =
      std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() -
                                                                t0)
          .count();

  if (status == "SOLVED") {
    auto grid = solver.GridAsList();
    if (VerifySolution(grid, col_hints, row_hints)) {
      return WorkerResult{pid, "SOLVED", std::nullopt, std::move(grid), elapsed};
    }
    return WorkerResult{pid, "UNSOLVED", std::nullopt, {}, elapsed};
  }

  if (status == "TIMEOUT") {
    return WorkerResult{pid, "TIMEOUT", payload, {}, elapsed};
  }
  return WorkerResult{pid, "UNSOLVED", std::nullopt, {}, elapsed};
}

struct Args {
  std::string input_file = "taai2019.txt";
  int max_workers = static_cast<int>(std::thread::hardware_concurrency());
  double slice_seconds = 30.0;
  int max_rounds = 10;
  bool single = false;
};

Args ParseArgs(int argc, char* argv[]) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input-file" && i + 1 < argc) {
      args.input_file = argv[++i];
    } else if (arg == "--max-workers" && i + 1 < argc) {
      args.max_workers = std::stoi(argv[++i]);
    } else if (arg == "--slice-seconds" && i + 1 < argc) {
      args.slice_seconds = std::stod(argv[++i]);
    } else if (arg == "--max-rounds" && i + 1 < argc) {
      args.max_rounds = std::stoi(argv[++i]);
    } else if (arg == "--single") {
      args.single = true;
    }
  }
  if (args.max_workers <= 0) {
    args.max_workers = 1;
  }
  return args;
}

}  // namespace

int main(int argc, char* argv[]) {
  Args args = ParseArgs(argc, argv);

  const std::string output_file = "result.txt";
  const std::string checkpoint_dir = "checkpoints";
  if (!EnsureCheckpointDir(checkpoint_dir)) {
    std::cerr << "Failed to create checkpoints directory.\n";
    return 1;
  }

  std::vector<Puzzle> puzzles = ReadPuzzles(args.input_file, args.single ? 1 : 0);
  if (puzzles.empty()) {
    std::cout << "No puzzles found.\n";
    return 0;
  }

  PrewarmPatterns(puzzles);

  std::unordered_map<std::string, double> total_times;
  std::unordered_map<std::string, SolverState> deferred;
  std::unordered_map<std::string, bool> solved;

  std::ofstream out(output_file);
  for (int round = 1; round <= args.max_rounds; ++round) {
    std::vector<std::tuple<std::string, std::vector<std::vector<int>>,
                           std::vector<std::vector<int>>, std::optional<SolverState>>>
        current;
    if (round == 1) {
      for (const auto& puzzle : puzzles) {
        std::optional<SolverState> state;
        std::string path = CheckpointPath(checkpoint_dir, puzzle.pid);
        auto loaded = ReadState(path);
        if (loaded.has_value()) {
          state = loaded;
        }
        current.emplace_back(puzzle.pid, puzzle.col_hints, puzzle.row_hints, state);
      }
    } else {
      for (const auto& [pid, state] : deferred) {
        auto it = std::find_if(puzzles.begin(), puzzles.end(),
                               [&](const Puzzle& puzzle) { return puzzle.pid == pid; });
        if (it != puzzles.end()) {
          current.emplace_back(pid, it->col_hints, it->row_hints, state);
        }
      }
    }

    if (current.empty()) {
      break;
    }

    deferred.clear();
    std::vector<std::future<WorkerResult>> futures;
    std::vector<WorkerResult> results;
    results.reserve(current.size());

    auto flush_one = [&]() {
      auto result = futures.front().get();
      futures.erase(futures.begin());
      results.push_back(std::move(result));
    };

    for (const auto& item : current) {
      if (futures.size() >= static_cast<std::size_t>(args.max_workers)) {
        flush_one();
      }
      futures.push_back(std::async(std::launch::async, SolveTimesliceWorker,
                                   std::get<0>(item), std::get<1>(item), std::get<2>(item),
                                   std::get<3>(item), args.slice_seconds));
    }
    while (!futures.empty()) {
      flush_one();
    }

    for (const auto& result : results) {
      total_times[result.pid] += result.elapsed;
      if (result.status == "SOLVED") {
        WriteSolution(out, result.pid, result.grid);
        std::cout << result.pid << " solved in " << std::fixed << std::setprecision(4)
                  << total_times[result.pid] << "s\n";
        solved[result.pid] = true;
        std::string path = CheckpointPath(checkpoint_dir, result.pid);
        std::remove(path.c_str());
      } else if (result.status == "TIMEOUT") {
        if (result.state.has_value()) {
          deferred[result.pid] = *result.state;
          std::string path = CheckpointPath(checkpoint_dir, result.pid);
          WriteState(path, *result.state);
        }
      } else {
        WriteUnsolved(out, result.pid);
        std::cout << result.pid << " unsolved in " << std::fixed << std::setprecision(4)
                  << total_times[result.pid] << "s\n";
        solved[result.pid] = true;
      }
    }
  }

  for (const auto& [pid, state] : deferred) {
    if (solved[pid]) {
      continue;
    }
    WriteUnsolved(out, pid);
    std::cout << pid << " unsolved in " << std::fixed << std::setprecision(4)
              << total_times[pid] << "s\n";
  }

  return 0;
}

/*
Self-test examples:
1) Single puzzle: ./solver --single
2) Full run: ./solver
3) Checkpoint resume: ./solver --slice-seconds 0.01
   then rerun with a larger slice to resume.
*/
