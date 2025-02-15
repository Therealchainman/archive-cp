# Leetcode Biweekly Rounds 150-199

# Leetcode Biweekly Contest 150

## 3453. Separate Squares I

### Solution 1:  binary search with tolerance, floating point numbers, greedy binary search

```cpp
class Solution {
private:
    vector<vector<int>> squares;
    int N;
    long double totalArea;
    bool possible(long double target) {
        long double area = 0;
        for (const vector<int>& square : squares) {
            int x = square[0], y = square[1], l = square[2];
            long double top = y + l;
            if (top <= target) {
                area += static_cast<long double>(l) * l;
            } else if (static_cast<long double>(y) <= target) {
                long double dy = target - y;
                area += dy * l;
            }
        }
        return area < (totalArea / 2.0);
    }
public:
    double separateSquares(vector<vector<int>>& sq) {
        squares = sq;
        N = squares.size();
        totalArea = 0;
        for (const vector<int>& square : squares) {
            int x = square[0], y = square[1], l = square[2];
            totalArea += static_cast<long double>(l) * l;
        }
        long double lo = 0, hi = 1e10;
        while (hi - lo > 1e-7) {
            long double mid = lo + (hi - lo) / 2.0;
            if (possible(mid)) lo = mid;
            else hi = mid;
        }
        return lo;
    }
};
```

## 3454. Separate Squares II

### Solution 1: line sweep, segment tree for union of rectangles, area covered by squares on 2d plane, segment tree with coordinate compression

1. segment tree calculates the width of each region covered by squares as you use line sweep over the y-value events. 
1. median y-coordinate, can find this with a little math, it is like interpolation

```cpp
using int64 = long long;
const int64 INF = 1e9;

struct SegmentTree {
    int N;
    vector<int64> count, total;
    vector<int64> xs;
    SegmentTree(vector<int64>& arr) {
        xs = vector<int64>(arr.begin(), arr.end());
        sort(xs.begin(), xs.end());
        xs.erase(unique(xs.begin(), xs.end()), xs.end());
        N = xs.size();
        count.assign(4 * N + 1, 0);
        total.assign(4 * N + 1, 0);
    }
    void update(int segmentIdx, int segmentLeftBound, int segmentRightBound, int64 l, int64 r, int64 val) {
        if (l >= r) return;
        if (l == xs[segmentLeftBound] && r == xs[segmentRightBound]) {
            count[segmentIdx] += val;
        } else {
            int mid = (segmentLeftBound + segmentRightBound) / 2;

            if (l < xs[mid]) {
                update(2 * segmentIdx, segmentLeftBound, mid, l, min(r, xs[mid]), val);
            }
            if (r > xs[mid]) {
                update(2 * segmentIdx + 1, mid, segmentRightBound, max(l, xs[mid]), r, val);
            }
        }
        if (count[segmentIdx] > 0) {
            total[segmentIdx] = xs[segmentRightBound] - xs[segmentLeftBound];
        } else {
            total[segmentIdx] = 2 * segmentIdx + 1 < total.size() ? total[2 * segmentIdx] + total[2 * segmentIdx + 1] : 0;
        }
    }
    void update(int l, int r, int val) {
        update(1, 0, N - 1, l, r, val);
    }
    int64 query() {
        return total[1];
    }
};

struct Event {
    int v, t, l, r;
    Event() {}
    Event(int v, int t, int l, int r) : v(v), t(t), l(l), r(r) {}
    bool operator<(const Event& other) const {
        if (v != other.v) return v < other.v;
        return t < other.t;
    }
};

class Solution {
public:
    double separateSquares(vector<vector<int>>& squares) {
        int N = squares.size();
        vector<int64> xs;
        for (const vector<int>& square : squares) {
            int x = square[0], y = square[1], l = square[2];
            xs.emplace_back(x);
            xs.emplace_back(x + l);
        }
        SegmentTree st(xs);
        vector<Event> events;
        for (const vector<int>& square : squares) {
            int x = square[0], y = square[1], l = square[2];
            events.emplace_back(y, 1, x, x + l);
            events.emplace_back(y + l, -1, x, x + l);
        }
        sort(events.begin(), events.end());
        int64 prevY = -INF;
        long double totalArea = 0;
        for (const Event& event : events) {
            long double dh = event.v - prevY;
            long double dw = st.query();
            totalArea += dh * dw;
            st.update(event.l, event.r, event.t);
            prevY = event.v;
        }
        prevY = -INF;
        long double curSumArea = 0;
        for (const Event& event : events) {
            long double dh = event.v - prevY;
            long double dw = st.query();
            long double area = dh * dw;
            if (2 * (area + curSumArea) >= totalArea) {
                return (totalArea / 2.0 - curSumArea) / dw + prevY;
            }
            curSumArea += area;
            st.update(event.l, event.r, event.t);
            prevY = event.v;
        }
        return curSumArea;
    }
};
```

## 3455. Shortest Matching Substring

### Solution 1:  kmp, string matching, dynamic programming, 

```cpp
const int INF = 1e9;
class Solution {
private:
    vector<int> kmp(const string& s) {
        int N = s.size();
        vector<int> pi(N, 0);
        for (int i = 1; i < N; i++) {
            int j = pi[i - 1];
            while (j > 0 && s[i] != s[j]) {
                j = pi[j - 1];
            }
            if (s[j] == s[i]) j++;
            pi[i] = j;
        }
        return pi;
    }
    vector<string> process(const string& s, char delimiter = ' ') {
        vector<string> ans;
        istringstream iss(s);
        string word;
        while (getline(iss, word, delimiter)) ans.emplace_back(word);
        return ans;
    }
public:
    int shortestMatchingSubstring(string s, string p) {
        int N = s.size(), M = p.size();
        vector<string> patterns = process(p, '*');
        while (patterns.size() < 3) {
            patterns.emplace_back("");
        }
        vector<vector<int>> pi(3);
        for (int i = 0; i < 3; i++) {
            vector<int> ret = kmp(patterns[i] + "#" + s);
            pi[i] = vector<int>(ret.begin() + patterns[i].size(), ret.end());
        }
        vector<vector<int>> dp(4, vector<int>(N + 1, -INF));
        iota(dp[0].begin(), dp[0].end(), 0);
        for (int i = 0; i < 3; i++) {
            for (int j = 1; j <= N; j++) {
                if (pi[i][j] == patterns[i].size()) {
                    dp[i + 1][j] = dp[i][j - pi[i][j]];
                }
                dp[i + 1][j] = max(dp[i + 1][j], dp[i + 1][j - 1]);
            }
        }
        int ans = INF;
        for (int i = 0; i <= N; i++) {
            if (dp[3][i] != -INF) {
                ans = min(ans, i - dp[3][i]);
            }
        }
        
        return ans < INF ? ans : -1;
    }
};
```

# Leetcode Biweekly Contest 151

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