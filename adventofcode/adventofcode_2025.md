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

### Part 2: k pointers

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

### Part 1: 

```cpp

```

### Part 2: 

```cpp

```

## Day 4

### Part 1: 

```cpp

```

### Part 2: 

```cpp

```

