#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

static const int UNKNOWN = -1;
static const int EMPTY = 0;
static const int FILLED = 1;

static size_t PATTERN_CACHE_SIZE = 20000;
static size_t CONSISTENT_CACHE_SIZE = 50000;

static int get_env_int(const char* name, int defv) {
    const char* v = std::getenv(name);
    if (!v) return defv;
    try {
        return std::stoi(v);
    } catch (...) {
        return defv;
    }
}

struct Key {
    int n;
    vector<int> hints;
    int must1;
    int must0;

    bool operator==(const Key& other) const {
        return n == other.n && must1 == other.must1 && must0 == other.must0 && hints == other.hints;
    }
};

struct KeyHash {
    size_t operator()(Key const& k) const {
        size_t h = std::hash<int>{}(k.n);
        h ^= std::hash<int>{}(k.must1 + 0x9e3779b9 + (h << 6) + (h >> 2));
        h ^= std::hash<int>{}(k.must0 + 0x9e3779b9 + (h << 6) + (h >> 2));
        for (auto v : k.hints) {
            h ^= std::hash<int>{}(v + 0x9e3779b9 + (h << 6) + (h >> 2));
        }
        return h;
    }
};

struct PatternKey {
    int n;
    vector<int> hints;
    bool operator==(const PatternKey& o) const {
        return n == o.n && hints == o.hints;
    }
};

struct PatternKeyHash {
    size_t operator()(PatternKey const& k) const {
        size_t h = std::hash<int>{}(k.n);
        for (auto v : k.hints) {
            h ^= std::hash<int>{}(v + 0x9e3779b9 + (h << 6) + (h >> 2));
        }
        return h;
    }
};

template <typename K, typename V, typename Hash = std::hash<K>>
class LRUCache {
public:
    LRUCache(size_t cap) : capacity(cap) {}

    bool get(const K& key, V& out) {
        auto it = mp.find(key);
        if (it == mp.end()) return false;
        lst.splice(lst.begin(), lst, it->second);
        out = it->second->second;
        return true;
    }

    void put(const K& key, const V& val) {
        auto it = mp.find(key);
        if (it != mp.end()) {
            it->second->second = val;
            lst.splice(lst.begin(), lst, it->second);
            return;
        }
        if (lst.size() >= capacity) {
            auto last = lst.back();
            mp.erase(last.first);
            lst.pop_back();
        }
        lst.emplace_front(key, val);
        mp[key] = lst.begin();
    }

private:
    size_t capacity;
    list<pair<K, V>> lst;
    unordered_map<K, typename list<pair<K, V>>::iterator, Hash> mp;
};

static LRUCache<Key, vector<int>, KeyHash> CONSISTENT_CACHE(50000);
static LRUCache<PatternKey, vector<int>, PatternKeyHash> PATTERN_CACHE(20000);

static vector<int> patterns_for_hints(int n, const vector<int>& hints) {
    PatternKey pk{n, hints};
    vector<int> cached;
    if (PATTERN_CACHE.get(pk, cached)) return cached;

    if (hints.empty()) {
        vector<int> res{0};
        PATTERN_CACHE.put(pk, res);
        return res;
    }

    int m = static_cast<int>(hints.size());
    vector<int> suffix_sum(m + 1, 0);
    for (int i = m - 1; i >= 0; --i) {
        suffix_sum[i] = suffix_sum[i + 1] + hints[i];
    }

    vector<int> res;

    function<void(int, int, int)> rec = [&](int pos, int i, int mask) {
        if (i == m) {
            res.push_back(mask);
            return;
        }
        int block_len = hints[i];
        int rem_blocks = m - (i + 1);
        int rem_min = suffix_sum[i + 1] + (rem_blocks > 0 ? rem_blocks : 0);
        int latest_start = n - (block_len + rem_min);
        if (latest_start < pos) return;

        int block_bits = (1 << block_len) - 1;
        for (int start = pos; start <= latest_start; ++start) {
            int new_mask = mask | (block_bits << start);
            int next_pos = start + block_len;
            if (i + 1 < m) next_pos += 1;
            rec(next_pos, i + 1, new_mask);
        }
    };

    rec(0, 0, 0);
    PATTERN_CACHE.put(pk, res);
    return res;
}

static vector<int> consistent_masks_cached(int n, const vector<int>& hints, int must1, int must0) {
    Key key{n, hints, must1, must0};
    vector<int> cached;
    if (CONSISTENT_CACHE.get(key, cached)) return cached;

    int fullmask = (1 << n) - 1;
    auto patterns = patterns_for_hints(n, hints);

    vector<int> out;
    out.reserve(patterns.size());
    for (int mask : patterns) {
        if ((mask & must0) != 0) continue;
        if ((mask & must1) != must1) continue;
        out.push_back(mask);
    }

    CONSISTENT_CACHE.put(key, out);
    return out;
}

static vector<int> hints_from_mask(int n, int mask) {
    vector<int> out;
    int run = 0;
    for (int i = 0; i < n; ++i) {
        if (mask & (1 << i)) {
            run++;
        } else if (run) {
            out.push_back(run);
            run = 0;
        }
    }
    if (run) out.push_back(run);
    return out;
}

class NonogramSolver {
public:
    struct TrailEntry {
        int r, c;
        int row1, row0, col1, col0;
    };

    struct Frame {
        int r;
        int c;
        vector<int> choices;
        int next_idx;
        int trail_len;
    };

    struct State {
        vector<int> row_must1;
        vector<int> row_must0;
        vector<int> col_must1;
        vector<int> col_must0;
        vector<TrailEntry> trail;
        vector<Frame> stack;
        bool initialized = false;
    };

    NonogramSolver(const vector<vector<int>>& col_hints, const vector<vector<int>>& row_hints)
        : row_hints(row_hints), col_hints(col_hints) {
        height = static_cast<int>(row_hints.size());
        width = static_cast<int>(col_hints.size());
        row_fullmask = (1 << width) - 1;
        col_fullmask = (1 << height) - 1;
        row_must1.assign(height, 0);
        row_must0.assign(height, 0);
        col_must1.assign(width, 0);
        col_must0.assign(width, 0);
        initialized = false;
    }

    State export_state() const {
        State state;
        state.row_must1 = row_must1;
        state.row_must0 = row_must0;
        state.col_must1 = col_must1;
        state.col_must0 = col_must0;
        state.trail = trail;
        state.stack = stack;
        state.initialized = initialized;
        return state;
    }

    void load_state(const State& state) {
        row_must1 = state.row_must1;
        row_must0 = state.row_must0;
        col_must1 = state.col_must1;
        col_must0 = state.col_must0;
        trail = state.trail;
        stack = state.stack;
        initialized = state.initialized;
    }

    bool is_solved() const {
        for (int r = 0; r < height; ++r) {
            if ((row_must1[r] | row_must0[r]) != row_fullmask) return false;
        }
        return true;
    }

    bool set_cell(int r, int c, int v, vector<TrailEntry>& trail_ref) {
        int bitc = 1 << c;
        int bitr = 1 << r;
        int row1 = row_must1[r];
        int row0 = row_must0[r];
        int col1 = col_must1[c];
        int col0 = col_must0[c];

        if (v == FILLED) {
            if ((row0 & bitc) || (col0 & bitr)) return false;
            if ((row1 & bitc) && (col1 & bitr)) return true;
            trail_ref.push_back({r, c, row1, row0, col1, col0});
            row_must1[r] = row1 | bitc;
            col_must1[c] = col1 | bitr;
            return true;
        }

        if ((row1 & bitc) || (col1 & bitr)) return false;
        if ((row0 & bitc) && (col0 & bitr)) return true;
        trail_ref.push_back({r, c, row1, row0, col1, col0});
        row_must0[r] = row0 | bitc;
        col_must0[c] = col0 | bitr;
        return true;
    }

    void undo_to(int trail_len) {
        while (static_cast<int>(trail.size()) > trail_len) {
            auto t = trail.back();
            trail.pop_back();
            row_must1[t.r] = t.row1;
            row_must0[t.r] = t.row0;
            col_must1[t.c] = t.col1;
            col_must0[t.c] = t.col0;
        }
    }

    bool solve_line_masks(int n, const vector<int>& hints, int must1, int must0, int fullmask,
                          int& out1, int& out0) {
        if (must1 & must0) return false;

        if ((must1 | must0) == fullmask) {
            if (hints_from_mask(n, must1) != hints) return false;
            out1 = must1;
            out0 = must0;
            return true;
        }

        auto masks = consistent_masks_cached(n, hints, must1, must0);
        if (masks.empty()) return false;

        int common1 = fullmask;
        int common0 = fullmask;
        for (int m : masks) {
            common1 &= m;
            common0 &= (~m) & fullmask;
        }

        int new_must1 = must1 | common1;
        int new_must0 = must0 | common0;
        if (new_must1 & new_must0) return false;

        out1 = new_must1;
        out0 = new_must0;
        return true;
    }

    bool propagate(vector<TrailEntry>& trail_ref, deque<pair<bool, int>>* qptr,
                   vector<bool>* in_row_ptr, vector<bool>* in_col_ptr) {
        deque<pair<bool, int>> q;
        vector<bool> in_row(height, false), in_col(width, false);
        if (!qptr) {
            for (int r = 0; r < height; ++r) {
                q.push_back({true, r});
                in_row[r] = true;
            }
            for (int c = 0; c < width; ++c) {
                q.push_back({false, c});
                in_col[c] = true;
            }
        } else {
            q = std::move(*qptr);
            in_row = std::move(*in_row_ptr);
            in_col = std::move(*in_col_ptr);
        }

        while (!q.empty()) {
            auto [is_row, idx] = q.front();
            q.pop_front();

            if (is_row) {
                in_row[idx] = false;
                int must1 = row_must1[idx];
                int must0 = row_must0[idx];

                int new_must1, new_must0;
                if (!solve_line_masks(width, row_hints[idx], must1, must0, row_fullmask, new_must1,
                                      new_must0))
                    return false;

                int delta1 = new_must1 & ~must1;
                int delta0 = new_must0 & ~must0;
                while (delta1) {
                    int bit = delta1 & -delta1;
                    int c = __builtin_ctz(bit);
                    if (!set_cell(idx, c, FILLED, trail_ref)) return false;
                    if (!in_col[c]) {
                        q.push_back({false, c});
                        in_col[c] = true;
                    }
                    delta1 &= delta1 - 1;
                }
                while (delta0) {
                    int bit = delta0 & -delta0;
                    int c = __builtin_ctz(bit);
                    if (!set_cell(idx, c, EMPTY, trail_ref)) return false;
                    if (!in_col[c]) {
                        q.push_back({false, c});
                        in_col[c] = true;
                    }
                    delta0 &= delta0 - 1;
                }
            } else {
                in_col[idx] = false;
                int must1 = col_must1[idx];
                int must0 = col_must0[idx];

                int new_must1, new_must0;
                if (!solve_line_masks(height, col_hints[idx], must1, must0, col_fullmask, new_must1,
                                      new_must0))
                    return false;

                int delta1 = new_must1 & ~must1;
                int delta0 = new_must0 & ~must0;
                while (delta1) {
                    int bit = delta1 & -delta1;
                    int r = __builtin_ctz(bit);
                    if (!set_cell(r, idx, FILLED, trail_ref)) return false;
                    if (!in_row[r]) {
                        q.push_back({true, r});
                        in_row[r] = true;
                    }
                    delta1 &= delta1 - 1;
                }
                while (delta0) {
                    int bit = delta0 & -delta0;
                    int r = __builtin_ctz(bit);
                    if (!set_cell(r, idx, EMPTY, trail_ref)) return false;
                    if (!in_row[r]) {
                        q.push_back({true, r});
                        in_row[r] = true;
                    }
                    delta0 &= delta0 - 1;
                }
            }
        }
        return true;
    }

    tuple<int, int, int> choose_guess() {
        bool has = false;
        int best_count = 0;
        bool best_is_row = true;
        int best_idx = -1;
        vector<int> best_masks;
        int best_must1 = 0, best_must0 = 0;

        for (int r = 0; r < height; ++r) {
            int must1 = row_must1[r];
            int must0 = row_must0[r];
            if ((must1 | must0) == row_fullmask) continue;
            auto masks = consistent_masks_cached(width, row_hints[r], must1, must0);
            if (masks.empty()) return {-1, -1, -1};
            if (!has || static_cast<int>(masks.size()) < best_count) {
                has = true;
                best_count = static_cast<int>(masks.size());
                best_is_row = true;
                best_idx = r;
                best_masks = std::move(masks);
                best_must1 = must1;
                best_must0 = must0;
            }
        }

        for (int c = 0; c < width; ++c) {
            int must1 = col_must1[c];
            int must0 = col_must0[c];
            if ((must1 | must0) == col_fullmask) continue;
            auto masks = consistent_masks_cached(height, col_hints[c], must1, must0);
            if (masks.empty()) return {-1, -1, -1};
            if (!has || static_cast<int>(masks.size()) < best_count) {
                has = true;
                best_count = static_cast<int>(masks.size());
                best_is_row = false;
                best_idx = c;
                best_masks = std::move(masks);
                best_must1 = must1;
                best_must0 = must0;
            }
        }

        if (!has) return {-1, -1, -1};

        int total = static_cast<int>(best_masks.size());
        int n = best_is_row ? width : height;
        int fullmask = (1 << n) - 1;
        int unknown_mask = (~(best_must1 | best_must0)) & fullmask;

        int best_pos = -1;
        int best_balance = 0;
        int best_pref = FILLED;

        int temp = unknown_mask;
        while (temp) {
            int bit = temp & -temp;
            int pos = __builtin_ctz(bit);
            int cnt1 = 0;
            for (int m : best_masks) {
                if (m & bit) cnt1++;
            }
            if (0 < cnt1 && cnt1 < total) {
                int balance = abs(2 * cnt1 - total);
                if (best_pos == -1 || balance < best_balance) {
                    best_pos = pos;
                    best_balance = balance;
                    best_pref = (cnt1 * 2 >= total) ? FILLED : EMPTY;
                }
            }
            temp &= temp - 1;
        }

        if (best_pos == -1) {
            int bit = unknown_mask & -unknown_mask;
            best_pos = __builtin_ctz(bit);
            best_pref = FILLED;
        }

        int r, c;
        if (best_is_row) {
            r = best_idx;
            c = best_pos;
        } else {
            r = best_pos;
            c = best_idx;
        }
        return {r, c, best_pref};
    }

    bool try_choice(int r, int c, int v) {
        if (!set_cell(r, c, v, trail)) return false;
        deque<pair<bool, int>> q;
        vector<bool> in_row(height, false), in_col(width, false);
        q.push_back({true, r});
        in_row[r] = true;
        q.push_back({false, c});
        in_col[c] = true;
        return propagate(trail, &q, &in_row, &in_col);
    }

    bool apply_frame_choice(Frame& frame) {
        while (frame.next_idx < static_cast<int>(frame.choices.size())) {
            int v = frame.choices[frame.next_idx++];
            undo_to(frame.trail_len);
            if (try_choice(frame.r, frame.c, v)) return true;
        }
        undo_to(frame.trail_len);
        return false;
    }

    bool backtrack() {
        while (!stack.empty()) {
            auto& frame = stack.back();
            if (apply_frame_choice(frame)) return true;
            stack.pop_back();
        }
        return false;
    }

    tuple<string, vector<vector<int>>, State> solve_timeslice(double slice_seconds, int check_interval) {
        if (!initialized) {
            if (!propagate(trail, nullptr, nullptr, nullptr)) {
                return {"UNSOLVED", {}, State{}};
            }
            initialized = true;
            if (is_solved()) return {"SOLVED", grid_as_list(), State{}};
        }

        auto start = chrono::high_resolution_clock::now();
        int nodes = 0;
        while (true) {
            if ((nodes & (check_interval - 1)) == 0) {
                auto now = chrono::high_resolution_clock::now();
                double elapsed = chrono::duration<double>(now - start).count();
                if (elapsed >= slice_seconds) {
                    return {"TIMEOUT", {}, export_state()};
                }
            }

            if (is_solved()) return {"SOLVED", grid_as_list(), State{}};

            auto [r, c, pref] = choose_guess();
            if (r == -1 && c == -1) {
                if (!backtrack()) return {"UNSOLVED", {}, State{}};
                nodes++;
                continue;
            }
            Frame frame;
            frame.r = r;
            frame.c = c;
            frame.choices = {pref, (pref == FILLED ? EMPTY : FILLED)};
            frame.next_idx = 0;
            frame.trail_len = static_cast<int>(trail.size());
            stack.push_back(frame);
            if (!apply_frame_choice(stack.back())) {
                if (!backtrack()) return {"UNSOLVED", {}, State{}};
            }
            nodes++;
        }
    }

    vector<vector<int>> grid_as_list() const {
        vector<vector<int>> grid(height, vector<int>(width, 0));
        for (int r = 0; r < height; ++r) {
            int mask = row_must1[r];
            for (int c = 0; c < width; ++c) {
                grid[r][c] = (mask & (1 << c)) ? 1 : 0;
            }
        }
        return grid;
    }

private:
    vector<vector<int>> row_hints;
    vector<vector<int>> col_hints;
    int height = 0, width = 0;
    int row_fullmask = 0, col_fullmask = 0;
    vector<int> row_must1, row_must0, col_must1, col_must0;

    vector<TrailEntry> trail;
    vector<Frame> stack;
    bool initialized = false;
};

static bool verify_solution(const vector<vector<int>>& grid, const vector<vector<int>>& col_hints,
                            const vector<vector<int>>& row_hints) {
    if (grid.empty()) return false;
    int rows = static_cast<int>(grid.size());
    int cols = static_cast<int>(grid[0].size());

    auto get_hints = [&](const vector<int>& line) {
        vector<int> h;
        int count = 0;
        for (int v : line) {
            if (v == 1) count++;
            else if (count > 0) {
                h.push_back(count);
                count = 0;
            }
        }
        if (count > 0) h.push_back(count);
        return h;
    };

    for (int r = 0; r < rows; ++r) {
        if (get_hints(grid[r]) != row_hints[r]) return false;
    }
    for (int c = 0; c < cols; ++c) {
        vector<int> col_line(rows);
        for (int r = 0; r < rows; ++r) col_line[r] = grid[r][c];
        if (get_hints(col_line) != col_hints[c]) return false;
    }
    return true;
}

static vector<int> parse_line(const string& line) {
    vector<int> out;
    string tmp = line;
    for (auto& ch : tmp) {
        if (ch == '.') ch = ' ';
    }
    stringstream ss(tmp);
    int v;
    while (ss >> v) out.push_back(v);
    return out;
}

struct Puzzle {
    string pid;
    vector<vector<int>> col_hints;
    vector<vector<int>> row_hints;
};

static vector<Puzzle> read_puzzles(const string& input_file, int limit) {
    ifstream fin(input_file);
    vector<string> lines;
    string line;
    while (getline(fin, line)) {
        if (!line.empty()) lines.push_back(line);
    }

    vector<Puzzle> puzzles;
    int idx = 0;
    while (idx < static_cast<int>(lines.size())) {
        if (!lines[idx].empty() && lines[idx][0] == '$') {
            string pid = lines[idx++];
            vector<vector<int>> col_hints;
            vector<vector<int>> row_hints;
            for (int i = 0; i < 25; ++i) col_hints.push_back(parse_line(lines[idx++]));
            for (int i = 0; i < 25; ++i) row_hints.push_back(parse_line(lines[idx++]));

            puzzles.push_back({pid, col_hints, row_hints});
            if (limit > 0 && static_cast<int>(puzzles.size()) >= limit) break;
        } else {
            idx++;
        }
    }
    return puzzles;
}

static void prewarm_patterns(const vector<Puzzle>& puzzles) {
    for (auto& p : puzzles) {
        for (auto& h : p.col_hints) {
            patterns_for_hints(25, h);
        }
        for (auto& h : p.row_hints) {
            patterns_for_hints(25, h);
        }
    }
}

static void ensure_checkpoint_dir(const string& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
}

static string checkpoint_path(const string& checkpoint_dir, const string& pid) {
    return checkpoint_dir + "/" + pid + ".pkl";
}

static void write_vector(ofstream& out, const vector<int>& vec) {
    int32_t size = static_cast<int32_t>(vec.size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(int32_t));
    if (size > 0) {
        out.write(reinterpret_cast<const char*>(vec.data()), sizeof(int32_t) * size);
    }
}

static void read_vector(ifstream& in, vector<int>& vec) {
    int32_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(int32_t));
    vec.resize(size);
    if (size > 0) {
        in.read(reinterpret_cast<char*>(vec.data()), sizeof(int32_t) * size);
    }
}

static void write_state(ofstream& out, const NonogramSolver::State& state) {
    write_vector(out, state.row_must1);
    write_vector(out, state.row_must0);
    write_vector(out, state.col_must1);
    write_vector(out, state.col_must0);

    int32_t trail_size = static_cast<int32_t>(state.trail.size());
    out.write(reinterpret_cast<const char*>(&trail_size), sizeof(int32_t));
    for (const auto& t : state.trail) {
        out.write(reinterpret_cast<const char*>(&t), sizeof(NonogramSolver::TrailEntry));
    }

    int32_t stack_size = static_cast<int32_t>(state.stack.size());
    out.write(reinterpret_cast<const char*>(&stack_size), sizeof(int32_t));
    for (const auto& f : state.stack) {
        out.write(reinterpret_cast<const char*>(&f.r), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&f.c), sizeof(int32_t));
        write_vector(out, f.choices);
        out.write(reinterpret_cast<const char*>(&f.next_idx), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&f.trail_len), sizeof(int32_t));
    }

    uint8_t init = state.initialized ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&init), sizeof(uint8_t));
}

static bool read_state(ifstream& in, NonogramSolver::State& state) {
    if (!in.good()) return false;
    read_vector(in, state.row_must1);
    read_vector(in, state.row_must0);
    read_vector(in, state.col_must1);
    read_vector(in, state.col_must0);

    int32_t trail_size = 0;
    in.read(reinterpret_cast<char*>(&trail_size), sizeof(int32_t));
    state.trail.resize(trail_size);
    for (int i = 0; i < trail_size; ++i) {
        in.read(reinterpret_cast<char*>(&state.trail[i]), sizeof(NonogramSolver::TrailEntry));
    }

    int32_t stack_size = 0;
    in.read(reinterpret_cast<char*>(&stack_size), sizeof(int32_t));
    state.stack.resize(stack_size);
    for (int i = 0; i < stack_size; ++i) {
        int32_t r, c, next_idx, trail_len;
        vector<int> choices;
        in.read(reinterpret_cast<char*>(&r), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&c), sizeof(int32_t));
        read_vector(in, choices);
        in.read(reinterpret_cast<char*>(&next_idx), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&trail_len), sizeof(int32_t));
        state.stack[i] = {r, c, choices, next_idx, trail_len};
    }

    uint8_t init = 0;
    in.read(reinterpret_cast<char*>(&init), sizeof(uint8_t));
    state.initialized = (init != 0);
    return in.good();
}

class ThreadPool {
public:
    explicit ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    function<void()> task;
                    {
                        unique_lock<mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template <class F>
    auto enqueue(F&& f) -> future<decltype(f())> {
        using Ret = decltype(f());
        auto task = make_shared<packaged_task<Ret()>>(std::forward<F>(f));
        future<Ret> res = task->get_future();
        {
            unique_lock<mutex> lock(queue_mutex);
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            unique_lock<mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& w : workers) w.join();
    }

private:
    vector<thread> workers;
    queue<function<void()>> tasks;
    mutex queue_mutex;
    condition_variable condition;
    bool stop;
};

struct WorkerResult {
    string pid;
    string status;
    vector<vector<int>> grid;
    NonogramSolver::State state;
    double elapsed = 0.0;
};

static WorkerResult solve_timeslice_worker(const Puzzle& puzzle, const NonogramSolver::State* state,
                                           double slice_seconds) {
    NonogramSolver solver(puzzle.col_hints, puzzle.row_hints);
    if (state) solver.load_state(*state);

    auto t0 = chrono::high_resolution_clock::now();
    auto [status, grid, new_state] = solver.solve_timeslice(slice_seconds, 512);
    double elapsed = chrono::duration<double>(chrono::high_resolution_clock::now() - t0).count();

    if (status == "SOLVED") {
        if (verify_solution(grid, puzzle.col_hints, puzzle.row_hints)) {
            return {puzzle.pid, "SOLVED", grid, NonogramSolver::State{}, elapsed};
        }
        return {puzzle.pid, "UNSOLVED", {}, NonogramSolver::State{}, elapsed};
    }

    if (status == "TIMEOUT") {
        return {puzzle.pid, "TIMEOUT", {}, new_state, elapsed};
    }

    return {puzzle.pid, "UNSOLVED", {}, NonogramSolver::State{}, elapsed};
}

static void write_solution(ofstream& out, const string& pid, const vector<vector<int>>& grid) {
    out << pid << "\n";
    for (const auto& row : grid) {
        for (size_t i = 0; i < row.size(); ++i) {
            out << (row[i] == FILLED ? "1" : "0");
            if (i + 1 < row.size()) out << "  ";
        }
        out << "\n";
    }
    out.flush();
}

static void write_unsolved(ofstream& out, const string& pid) {
    out << pid << "\n";
    out << "Unsolvable or Timeout\n";
    out.flush();
}

int main(int argc, char** argv) {
    PATTERN_CACHE_SIZE = static_cast<size_t>(get_env_int("NONO_PATTERN_CACHE", 20000));
    CONSISTENT_CACHE_SIZE = static_cast<size_t>(get_env_int("NONO_CONSISTENT_CACHE", 50000));
    PATTERN_CACHE = LRUCache<PatternKey, vector<int>, PatternKeyHash>(PATTERN_CACHE_SIZE);
    CONSISTENT_CACHE = LRUCache<Key, vector<int>, KeyHash>(CONSISTENT_CACHE_SIZE);

    string input_file = "taai2019.txt";
    int max_workers = static_cast<int>(thread::hardware_concurrency());
    double slice_seconds = 30.0;
    int max_rounds = 10;
    bool single = false;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--input-file" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "--max-workers" && i + 1 < argc) max_workers = stoi(argv[++i]);
        else if (arg == "--slice-seconds" && i + 1 < argc) slice_seconds = stod(argv[++i]);
        else if (arg == "--max-rounds" && i + 1 < argc) max_rounds = stoi(argv[++i]);
        else if (arg == "--single") single = true;
    }

    if (max_workers < 1) max_workers = 1;

    auto puzzles = read_puzzles(input_file, single ? 1 : 0);
    if (puzzles.empty()) {
        cout << "No puzzles found.\n";
        return 0;
    }

    prewarm_patterns(puzzles);

    string output_file = "result_py.txt";
    string checkpoint_dir = "checkpoints";
    ensure_checkpoint_dir(checkpoint_dir);

    unordered_map<string, NonogramSolver::State> deferred;
    unordered_map<string, double> total_times;
    unordered_map<string, Puzzle> puzzle_map;
    for (const auto& p : puzzles) puzzle_map[p.pid] = p;

    ofstream fout(output_file);
    cout.setf(ios::fixed);
    cout << setprecision(4);

    for (int round_idx = 1; round_idx <= max_rounds; ++round_idx) {
        vector<pair<string, NonogramSolver::State*>> current;
        if (round_idx == 1) {
            for (auto& puzzle : puzzles) {
                string path = checkpoint_path(checkpoint_dir, puzzle.pid);
                ifstream fin(path, ios::binary);
                if (fin.good()) {
                    NonogramSolver::State state;
                    if (read_state(fin, state)) {
                        deferred[puzzle.pid] = state;
                    }
                }
                auto it = deferred.find(puzzle.pid);
                if (it != deferred.end()) {
                    current.push_back({puzzle.pid, &it->second});
                } else {
                    current.push_back({puzzle.pid, nullptr});
                }
            }
        } else {
            for (auto& kv : deferred) {
                current.push_back({kv.first, &kv.second});
            }
        }

        if (current.empty()) break;
        deferred.clear();

        ThreadPool pool(static_cast<size_t>(max_workers));
        vector<future<WorkerResult>> futures;
        futures.reserve(current.size());
        for (auto& item : current) {
            const auto& pid = item.first;
            auto* state = item.second;
            futures.push_back(pool.enqueue([pid, state, &puzzle_map, slice_seconds]() {
                return solve_timeslice_worker(puzzle_map.at(pid), state, slice_seconds);
            }));
        }

        for (auto& fut : futures) {
            auto result = fut.get();
            total_times[result.pid] += result.elapsed;
            if (result.status == "SOLVED") {
                write_solution(fout, result.pid, result.grid);
                cout << result.pid << " solved in " << total_times[result.pid] << "s\n";
                string path = checkpoint_path(checkpoint_dir, result.pid);
                std::remove(path.c_str());
            } else if (result.status == "TIMEOUT") {
                deferred[result.pid] = result.state;
                string path = checkpoint_path(checkpoint_dir, result.pid);
                ofstream fout_cp(path, ios::binary);
                write_state(fout_cp, result.state);
            } else {
                write_unsolved(fout, result.pid);
                cout << result.pid << " unsolved in " << total_times[result.pid] << "s\n";
            }
        }
    }

    for (auto& kv : deferred) {
        write_unsolved(fout, kv.first);
        cout << kv.first << " unsolved in " << total_times[kv.first] << "s\n";
    }

    return 0;
}
