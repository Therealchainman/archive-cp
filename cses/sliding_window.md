# Sliding Window Problems

## Sliding Window Sum

### Solution 1: queue, maintain current window sum, invertible operation

```cpp
int N, K, X, A, B, C;

void solve() {
    cin >> N >> K >> X >> A >> B >> C;
    int val = X;
    int64 w = 0, ans = 0;
    queue<int> q;
    for (int i = 0; i < N; ++i) {
        q.emplace(val);
        w += val;
        if (i + 1 >= K) {
            ans ^= w;
            w -= q.front();
            q.pop();
        }
        val = (1LL * val * A + B) % C;
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Sliding Window Minimum

### Solution 1: monotonic deque to maintain current window minimum, weakly increasing in deque, first element is always minimum for window

```cpp
int N, K, X, A, B, C;

void solve() {
    cin >> N >> K >> X >> A >> B >> C;
    int val = X, w = 0, ans = 0;
    deque<pair<int, int>> dq;
    for (int i = 0; i < N; ++i) {
        while (!dq.empty() && dq.back().first >= val) dq.pop_back();
        dq.emplace_back(val, i);
        if (i + 1 >= K) {
            ans ^= dq.front().first;
            if (dq.front().second == i - K + 1) dq.pop_front();
        }
        val = (1LL * val * A + B) % C;
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```
## Sliding Window Xor

### Solution 1: queue, maintain current window xor, invertible operation

```cpp
int N, K, X, A, B, C;

void solve() {
    cin >> N >> K >> X >> A >> B >> C;
    int val = X, w = 0, ans = 0;
    queue<int> q;
    for (int i = 0; i < N; ++i) {
        q.emplace(val);
        w ^= val;
        if (i + 1 >= K) {
            ans ^= w;
            w ^= q.front();
            q.pop();
        }
        val = (1LL * val * A + B) % C;
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Sliding Window Or

### Solution 1: two stack approach, prefix and suffix stack trick for sliding window, invertible operation

stacks 

unlike sliding window for sum, you can't just remove the left element from the window when it goes out of range, because or is not invertible.

This trick works for calculating gcd, lcm, AND, OR over a sliding window.

```cpp
int N, K, X, A, B, C;

int OR(const stack<int> &stk) {
    return !stk.empty() ? stk.top() : 0;
}

void solve() {
    cin >> N >> K >> X >> A >> B >> C;
    stack<int> inVal, inAgg, outVal, outAgg;
    int val = X, ans = 0;
    for (int i = 0; i < N; ++i) {
        inVal.emplace(val);
        inAgg.emplace(OR(inAgg) | val);
        if (inVal.size() == K) {
            while (!inVal.empty()) {
                int c = inVal.top();
                inVal.pop();
                inAgg.pop();
                outVal.emplace(c);
                outAgg.emplace(OR(outAgg) | c);
            }
        }
        if (i + 1 >= K) {
            int cand = OR(inAgg) | OR(outAgg);
            ans ^= cand;
        }
        // remove element
        if (!outVal.empty()) {
            outVal.pop();
            outAgg.pop();
        }
        val = (1LL * val * A + B) % C;
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Sliding Window Distinct Values

### Solution 1: frequency map, invertible sum

```cpp
int N, K;
vector<int> A;

void solve() {
    cin >> N >> K;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    int w = 0;
    map<int, int> freq;
    for (int i = 0; i < N; ++i) {
        freq[A[i]]++;
        if (freq[A[i]] == 1) w++;
        if (i + 1 >= K) {
            cout << w << " ";
            freq[A[i - K + 1]]--;
            if (freq[A[i - K + 1]] == 0) w--;
        }
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Sliding Window Mode

### Solution 1: min/max heap to maintain highest frequency element in the window, includes tiebreaker rule for taking smallest element in case of frequency tie

```cpp
struct Point {
    int freq, val;
    Point() {};
    Point(int freq, int val) : freq(freq), val(val) {};
    bool operator<(const Point &other) const {
        if (freq != other.freq) return freq < other.freq;
        return val > other.val;
    }
};

int N, K;
vector<int> A;
priority_queue<Point> heap;
map<int, int> freq;

void solve() {
    cin >> N >> K;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    for (int i = 0; i < N; ++i) {
        freq[A[i]]++;
        heap.emplace(freq[A[i]], A[i]);
        while (!heap.empty() && heap.top().freq != freq[heap.top().val]) heap.pop();
        if (i + 1 >= K) {
            cout << heap.top().val << " ";
            freq[A[i - K + 1]]--;
        }
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Sliding Window Mex

### Solution 1: 

```cpp

```

## Sliding Window Median

### Solution 1: two multisets, maintain lower half and upper half of the window elements, balance after each insertion and deletion

```cpp
int N, K;
vector<int> A;

void solve() {
    cin >> N >> K;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    multiset<int> low, high;
    for (int i = 0; i < N; ++i) {
        low.emplace(A[i]);
        if (low.size() > high.size()) {
            high.emplace(*low.rbegin());
            low.erase(prev(low.end()));
        }
        // sometimes the element inserted in low should have been in high, this balances that
        if (!low.empty() && !high.empty() && *low.rbegin() > *high.begin()) {
            auto x = high.begin(), y = prev(low.end());
            low.emplace(*x);
            high.emplace(*y);
            high.erase(x);
            low.erase(y);
        }
        if (i + 1 >= K) {
            int ans = K & 1 ? *high.begin() : *low.rbegin();
            cout << ans << " ";
            int x = A[i - K + 1];
            if (low.find(x) != low.end()) {
                low.erase(low.find(x));
                low.emplace(*high.begin());
                high.erase(high.begin());
            } else {
                high.erase(high.find(x));
            }
        }
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Sliding Window Cost

### Solution 1: sliding window median, two multisets, track lower and upper half sums to compute cost efficiently, cost is sum of absolute differences to median

```cpp
int N, K;
vector<int> A;

void solve() {
    cin >> N >> K;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    multiset<int> low, high;
    int64 lowSum = 0, highSum = 0;
    for (int i = 0; i < N; ++i) {
        low.emplace(A[i]);
        lowSum += A[i];
        if (low.size() > high.size()) {
            high.emplace(*low.rbegin());
            lowSum -= *low.rbegin();
            highSum += *low.rbegin();
            low.erase(prev(low.end()));
        }
        // sometimes the element inserted in low should have been in high, this balances that
        if (!low.empty() && !high.empty() && *low.rbegin() > *high.begin()) {
            auto x = high.begin(), y = prev(low.end());
            low.emplace(*x);
            high.emplace(*y);
            lowSum = lowSum + *x - *y;
            highSum = highSum - *x + *y;
            high.erase(x);
            low.erase(y);
        }
        if (i + 1 >= K) {
            int med = K & 1 ? *high.begin() : *low.rbegin();
            int64 lowCand = 1LL * low.size() * med - lowSum;
            int64 highCand = highSum - 1LL * high.size() * med;
            int64 ans = lowCand + highCand;
            cout << ans << " ";
            int x = A[i - K + 1];
            if (low.find(x) != low.end()) {
                low.erase(low.find(x));
                low.emplace(*high.begin());
                lowSum = lowSum - x + *high.begin();
                highSum -= *high.begin();
                high.erase(high.begin());
            } else {
                high.erase(high.find(x));
                highSum -= x;
            }
        }
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Sliding Window Inversions

### Solution 1: 

```cpp

```

## Sliding Window Advertisement

### Solution 1: 

```cpp

```
