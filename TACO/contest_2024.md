# TACO 2024 Advanced

##

### Solution 1: 

```cpp

```

## D. Map Coloring

### Solution 1:  binary search, color points farthest from each other to be in cluster 1 and 3, then place everything else in cluster 2

```cpp
pair<int, int> max_manhattan_distance(const vector<pair<int, int>>& points) {
    int smin = INT_MAX, dmin = INT_MAX;
    int smax = INT_MIN, dmax = INT_MIN;
    int smax_i = -1, smin_i = -1, dmax_i = -1, dmin_i = -1;
    
    for (int i = 0; i < points.size(); ++i) {
        int x = points[i].first, y = points[i].second;
        int s = x + y;
        int d = x - y;
        if (s > smax) {
            smax = s;
            smax_i = i;
        }
        if (s < smin) {
            smin = s;
            smin_i = i;
        }
        if (d > dmax) {
            dmax = d;
            dmax_i = i;
        }
        if (d < dmin) {
            dmin = d;
            dmin_i = i;
        }
    }
    if (smax - smin >= dmax - dmin)
        return make_pair(smax_i, smin_i);
    else
        return make_pair(dmax_i, dmin_i);
}

int manhattan_distance(int x1, int y1, int x2, int y2) {
    return abs(x2 - x1) + abs(y2 - y1);
}

void solve() {
    int N;
    cin >> N;
    vector<pair<int, int>> pos(N);
    for (int i = 0; i < N; ++i) {
        cin >> pos[i].first >> pos[i].second;
    }

    pair<int, int> indices = max_manhattan_distance(pos);
    int p1 = indices.first;
    int p3 = indices.second;

    auto possible = [&pos, &p1, &p3](int target) {
        int x1 = pos[p1].first, y1 = pos[p1].second;
        int x3 = pos[p3].first, y3 = pos[p3].second;
        vector<pair<int, int>> p2;

        for (auto& [x, y] : pos) {
            if (manhattan_distance(x1, y1, x, y) > target && manhattan_distance(x3, y3, x, y) > target) {
                p2.push_back(make_pair(x, y));
            }
        }

        if (p2.empty()) return true;
        auto [z1, z2] = max_manhattan_distance(p2);
        return manhattan_distance(p2[z1].first, p2[z1].second, p2[z2].first, p2[z2].second) <= target;
    };

    int lo = 0, hi = 2e9 + 5;
    while (lo < hi) {
        int mi = (lo + hi) / 2;
        if (possible(mi)) {
            hi = mi;
        } else {
            lo = mi + 1;
        }
    }
    cout << lo << endl;
}

signed main() {
    solve();
    return 0;
}
```

## A. Yet Another No 7

### Solution 1:  digit dp, binary search, count number of integers that do not contain 7 for some x

```cpp
int count(int pos, bool tight, int num, vector<vector<int>>& dp, const vector<int>& digits) {
    if (pos == digits.size()) {
        return 1; // Found a valid number
    }

    if (dp[pos][tight] != -1) {
        return dp[pos][tight];
    }

    int limit = tight ? digits[pos] : 9;
    int res = 0;
    for (int dig = 0; dig <= limit; ++dig) {
        if (dig == 7) continue; // Skip if the digit is 7
        res += count(pos + 1, tight && (dig == limit), num * 10 + dig, dp, digits);
    }

    dp[pos][tight] = res;
    return res;
}

void solve() {
    int n, k;
    cin >> n >> k;
    int lo = 0, hi = 2e18;
    while (lo < hi) {
        int mi = lo + (hi - lo) / 2;
        string mi_str = to_string(mi);
        vector<int> digits(mi_str.length());
        for (int i = 0; i < mi_str.length(); ++i) {
            digits[i] = mi_str[i] - '0';
        }
        vector<vector<int>> dp(digits.size(), vector<int>(2, -1));
        int cnt = count(0, 1, 0, dp, digits) - 1;
        if (cnt < k) {
            lo = mi + 1;
        } else {
            hi = mi;
        }
    }
    cout << (lo <= n ? lo : -1) << endl;
}

signed main() {
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```