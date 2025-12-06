# Advent of Code 2025

## Day 1

### Part 1: modular arithmetic, circular indexing, 

```cpp
const int MOD = 100;

void solve() {
    int ans = 0;
    int pos = 50;
    string line;
    while (getline(cin, line)) {
        int step = stoi(line.substr(1));
        int r = step % MOD;
        pos = pos + (line[0] == 'R' ? r : -r);
        if (pos < 0) pos += MOD;
        pos %= MOD;
        if (pos == 0) ans++;
    }
    debug(ans, "\n");
}
```

### Part 2: modular arithmetic, remainders

1. step / MOD is the number of full circles you make so they visit 0. 
1. r is number of extra clicks that do not form a full circle. 
1. rotating left, you must check that you started above 0, pos > 0, and then you ended at or below 0, npos <= 0

```cpp
const int MOD = 100;

void solve() {
    int ans = 0;
    int pos = 50;
    string line;
    while (getline(cin, line)) {
        int step = stoi(line.substr(1));
        ans += step / MOD;
        int r = step % MOD;
        if (pos == 0 && line[0] == 'L') ans--; // edge case
        pos += (line[0] == 'R' ? r : -r);
        if (pos <= 0 || pos >= MOD) ans++;
        if (pos < 0) pos += MOD;
        if (pos >= MOD) pos -= MOD;
    }
    debug(ans, "\n");
}
```

## Day 2

### Part 1: two pointer, string processing

```cpp
vector<pair<int64, int64>> process(const string& s) {
    vector<pair<int64, int64>> ans;
    istringstream iss(s);
    string x;
    while (getline(iss, x, ',')) {
        int idx = x.find('-');
        int64 l = stoll(x.substr(0, idx)), r = stoll(x.substr(idx + 1));
        ans.emplace_back(l, r);
    }
    return ans;
}

bool isSatisfied(int64 x) {
    string cand = to_string(x);
    int N = cand.size();
    if (N & 1) return false;
    for (int l = 0, r = N / 2; l < N / 2; ++l, ++r) {
        if (cand[l] != cand[r]) return false;
    }
    return true;
}

void solve() {
    string line;
    cin >> line;
    vector<pair<int64, int64>> A = process(line);
    int64 ans = 0;
    for (const auto &[x, y] : A) {
        for (int64 i = x; i <= y; ++i) {
            if (isSatisfied(i)) {
                ans += i;
            }
        }
    }
    debug(ans, "\n");
}
```

### Part 2: split into blocks, check each position in all blocks are equal

I test every segment length i that divides the total number of digits. For each such i, imagine splitting the number into N / i blocks of length i. Then verify that each digit in a given position inside a block is the same across all blocks. If that holds for every position inside the block, the whole number is just a repetition of that block.

```cpp
vector<pair<int64, int64>> process(const string& s) {
    vector<pair<int64, int64>> ans;
    istringstream iss(s);
    string x;
    while (getline(iss, x, ',')) {
        int idx = x.find('-');
        int64 l = stoll(x.substr(0, idx)), r = stoll(x.substr(idx + 1));
        ans.emplace_back(l, r);
    }
    return ans;
}

bool isSatisfied(int64 x) {
    string cand = to_string(x);
    int N = cand.size();
    for (int i = 1; i < N; ++i) {
        if (N % i != 0) continue;
        // try size i
        bool isRepeating = true;
        for (int j = 0; j < i; ++j) {
            // take current dig
            char dig = cand[j];
            for (int k = j; k < N; k += i) {
                if (dig != cand[k]) {
                    isRepeating = false;
                    break;
                }
            }
            if (!isRepeating) break;
        }
        if (isRepeating) return true;
    }
    return false;
}

void solve() {
    string line;
    cin >> line;
    vector<pair<int64, int64>> A = process(line);
    int64 ans = 0;
    for (const auto &[x, y] : A) {
        for (int64 i = x; i <= y; ++i) {
            if (isSatisfied(i)) {
                ans += i;
            }
        }
    }
    debug(ans, "\n");
}
```

## Day 3

### Part 1: sliding window, greedy

```cpp
pair<int, int> findDigit(const string& s) {
    int dig = -1, idx = -1, N = s.size();
    for (int i = 0; i < N; ++i) {
        int cand = s[i] - '0';
        if (cand > dig) {
            dig = cand;
            idx = i;
        }
    }
    return {dig, idx};
}
int64 findValue(const string& s) {
    int N = s.size();
    auto [d1, i] = findDigit(s.substr(0, N - 1));
    auto [d2, j] = findDigit(s.substr(i + 1));
    return d1 * 10 + d2;
}

vector<string> banks;

void solve() {
    string line;
    while (getline(cin, line)) {
        banks.emplace_back(line);
    }
    int64 ans = 0;
    for (const string &s : banks) {
        ans += findValue(s);
    }
    debug(ans, "\n");
}
```

### Part 2: sliding window, greedy

```cpp
pair<int, int> findDigit(const string& s, int si, int ei) {
    int dig = -1, idx = -1, N = s.size();
    for (int i = si; i <= ei; ++i) {
        int cand = s[i] - '0';
        if (cand > dig) {
            dig = cand;
            idx = i;
        }
    }
    return {dig, idx};
}
int64 findValue(const string& s) {
    int N = s.size();
    int64 ans = 0;
    for (int i = N - 12, j = 0; i < N; ++i) {
        auto [d, k] = findDigit(s, j, i);
        ans = 10LL * ans + d;
        j = k + 1;
    }
    return ans;
}

void solve() {
    string line;
    int64 ans = 0;
    while (getline(cin, line)) {
        ans += findValue(line);
    }
    debug(ans, "\n");
}
```

## Day 4

### Part 1: grid, counting 8 neighbors

```cpp
const char EMPTY = '.', PAPER = '@';
vector<vector<char>> grid;
int N;
bool inBounds(int r, int c) {
    return r >= 0 && r < N && c >= 0 && c < N;
}
bool isAccessible(int r, int c) {
    int cnt = 0;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (abs(dr) + abs(dc) == 0) continue;
            int nr = r + dr, nc = c + dc;
            if (!inBounds(nr, nc)) continue;
            if (grid[nr][nc] == PAPER) cnt++;
        }
    }
    return cnt < 4;
}
void solve() {
    grid.clear();
    string line;
    while (getline(cin, line)) {
        // best way to convert a string into a vector<char>
        grid.emplace_back(vector<char>(line.begin(), line.end()));
    }
    N = grid.size();
    int ans = 0;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (grid[r][c] != PAPER) continue;
            if (isAccessible(r, c)) ans++;
        }
    }
    debug(ans, "\n");
}
```

### Part 2: queue, bfs, flood fill neighbors, grid

```cpp
const char EMPTY = '.', PAPER = '@';
vector<vector<char>> grid;
vector<vector<int>> C;
int N;
bool inBounds(int r, int c) {
    return r >= 0 && r < N && c >= 0 && c < N;
}
vector<pair<int ,int>> neighbors(int r, int c) {
    vector<pair<int, int>> ans;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (abs(dr) + abs(dc) == 0) continue;
            int nr = r + dr, nc = c + dc;
            if (!inBounds(nr, nc)) continue;
            ans.emplace_back(nr, nc);
        }
    }
    return ans;
}
int countNeighbors(int r, int c) {
    int cnt = 0;
    for (auto [nr, nc] : neighbors(r, c)) {
        if (grid[nr][nc] == PAPER) cnt++;
    }
    return cnt;
}
void solve() {
    grid.clear();
    C.clear();
    string line;
    while (getline(cin, line)) {
        grid.emplace_back(vector<char>(line.begin(), line.end()));
    }
    N = grid.size();
    C.assign(N, vector<int>(N, 0));
    queue<pair<int, int>> q;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (grid[r][c] == EMPTY) continue;
            C[r][c] = countNeighbors(r, c);
            if (C[r][c] < 4) q.emplace(r, c);
        }
    }
    int ans = 0;
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        ans++;
        for (auto [nr, nc] : neighbors(r, c)) {
            if (C[nr][nc]-- == 4) q.emplace(nr, nc);
        }
    }
    debug(ans, "\n");
}
```

## Day 5

### Part 1: point in interval, array, regex

```cpp
regex freshRegex(R"((\d+)-(\d+))");
vector<pair<int64, int64>> R;
vector<int64> arr;

bool inBounds(int64 l, int64 r, int64 x) {
    return x >= l && x <= r;
}
bool isFresh(int64 x) {
    for (auto [l, r] : R) {
        if (inBounds(l, r, x)) return true;
    }
    return false;
}
void solve() {
    R.clear(); arr.clear();
    string line;
    while (getline(cin, line)) {
        if (line.empty()) continue;
        smatch match;
        if (regex_match(line, match, freshRegex)) {
            int64 left = stoll(match[1]), right = stoll(match[2]);
            R.emplace_back(left, right);
        } else {
            arr.emplace_back(stoll(line));
        }
    }
    int ans = 0;
    for (int64 x : arr) {
        if (isFresh(x)) ans++;
    }
    debug(ans, "\n");
}
```

### Part 2: 1d intersection, overlapping, merge intervals, accumulate

```cpp
regex freshRegex(R"((\d+)-(\d+))");
vector<pair<int64, int64>> arr;

bool intersects(int64 s1, int64 e1, int64 s2, int64 e2) {
    int64 s = max(s1, s2), e = min(e1, e2);
    return e >= s;
}
void solve() {
    arr.clear();
    string line;
    while (getline(cin, line)) {
        if (line.empty()) break;
        smatch match;
        if (regex_match(line, match, freshRegex)) {
            int64 left = stoll(match[1]), right = stoll(match[2]);
            arr.emplace_back(left, right);
        }
    }
    sort(arr.begin(), arr.end());
    vector<pair<int64, int64>> merged;
    for (auto [l, r] : arr) {
        if (merged.empty()) {
            merged.emplace_back(l, r);
            continue;
        }
        auto [pl, pr] = merged.back();
        if (intersects(pl, pr, l, r)) {
            merged.pop_back();
            merged.emplace_back(min(pl, l), max(pr, r));
        } else {
            merged.emplace_back(l, r);
        }
    }
    int64 ans = accumulate(merged.begin(), merged.end(), 0LL, [](int64 accum, const pair<int64, int64> x) {
        return accum + x.second - x.first + 1;
    });
    debug(ans, "\n");
}
```

## Day 6

### Part 1: string, arithmetic

```cpp
const string ADDITION = "+", MULTIPLICATION = "*";
vector<vector<string>> grid;
vector<string> getRow(const string& s) {
    vector<string> res;
    istringstream iss(s);
    string col;
    while (iss >> col) {
        res.emplace_back(col);
    }
    return res;
}
void solve() {
    grid.clear();
    string line;
    while (getline(cin, line)) {
        vector<string> row = getRow(line);
        grid.emplace_back(row);
    }
    int R = grid.size(), C = grid[0].size();
    int64 ans = 0;
    for (int c = 0; c < C; ++c) {
        string op = grid[R - 1][c];
        int64 val = op == ADDITION ? 0 : 1;
        for (int r = 0; r + 1 < R; ++r) {
            if (op == ADDITION) val += stoll(grid[r][c]);
            else val *= stoll(grid[r][c]);
        }
        ans += val;
    }
    debug(ans, "\n");
}
```

### Part 2: string, map, grid, arithmetic

```cpp
const char ADDITION = '+', MULTIPLICATION = '*';
int getDigit(char c) {
    return c - '0';
}
vector<char> getOperations(const string& s) {
    vector<char> res;
    for (const char c : s) {
        if (c == ADDITION || c == MULTIPLICATION) res.emplace_back(c);
    }
    return res;
}
void solve() {
    vector<char> operations;
    map<int, int64> colVal;
    string line;
    while (getline(cin, line)) {
        for (int i = 0; i < line.size(); ++i) {
            if (isdigit(line[i])) colVal[i] = colVal[i] * 10 + getDigit(line[i]);
        }
        operations = getOperations(line); // works but you know
    }
    int N = operations.size();
    vector<vector<int64>> A;
    int64 ans = 0;
    for (int i = 0, j = 0; i < N; ++j, ++i) {
        vector<int64> row;
        while (colVal.find(j) != colVal.end()) {
            row.emplace_back(colVal[j++]);
        }
        A.emplace_back(row);
    }
    assert(A.size() == N);
    for (int i = 0; i < N; ++i) {
        int64 val = operations[i] == ADDITION ? 0 : 1;
        for (int64 x : A[i]) {
            if (operations[i] == ADDITION) val += x;
            else val *= x;
        }
        ans += val;
    }
    debug(ans, "\n");
}
```

## Day 7

### Part 1: 

```cpp

```

### Part 2: 

```cpp

```

## Day 8

### Part 1: 

```cpp

```

### Part 2: 

```cpp

```

