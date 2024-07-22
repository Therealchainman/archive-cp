# Teamscode Spring 2024 (Advanced Division)

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
const int INF = 1e9;
int N, Q;
vector<int> arr;

struct Node {
    int mx, cnt, idx;
};

struct SegmentTree {
    int size;
    vector<int> maximum, counts, index;

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        maximum.assign(2 * size, -INF);
        counts.assign(2 * size, 0);
        index.assign(2 * size, -1);
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            int left_max = maximum[left_segment_idx], right_max = maximum[right_segment_idx];
            int left_count = counts[left_segment_idx], right_count = counts[right_segment_idx];
            int left_idx = index[left_segment_idx], right_idx = index[right_segment_idx];
            maximum[segment_idx] = max(left_max, right_max);
            counts[segment_idx] = 0;
            if (left_max == maximum[segment_idx]) counts[segment_idx] ^= left_count;
            if (right_max == maximum[segment_idx]) counts[segment_idx] ^= right_count;
            if (left_max == maximum[segment_idx]) index[segment_idx] = left_idx;
            if (right_max == maximum[segment_idx]) index[segment_idx] = right_idx; // rightmost index, cause it may be the one not cancelled
            segment_idx >>= 1; 
        }
    }

    void update(int segment_idx, int idx, int val) {
        segment_idx += size;
        maximum[segment_idx] = val;
        counts[segment_idx] = 1;
        index[segment_idx] = idx;
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    Node query(int left, int right) {
        left += size, right += size;
        Node res = {-INF, 0, -1};
        while (left <= right) {
            if (left & 1) {
                if (maximum[left] > res.mx) {
                    res.mx = maximum[left];
                    res.cnt = counts[left];
                    res.idx = index[left];
                } else if (maximum[left] == res.mx) {
                    res.cnt += counts[left];
                    res.idx = max(res.idx, index[left]);
                }
                left++;
            }
            if (~right & 1) {
                if (maximum[right] > res.mx) {
                    res.mx = maximum[right];
                    res.cnt = counts[right];
                    res.idx = index[right];
                } else if (maximum[right] == res.mx) {
                    res.cnt += counts[right];
                    res.idx = max(res.idx, index[right]);
                }
                right--;
            }
            left >>= 1, right >>= 1;
        }
        return res;
    }
};

void solve() {
    cin >> N >> Q;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    SegmentTree seg;
    seg.init(N);
    for (int i = 0; i < N; i++) {
        seg.update(i, i, arr[i]);
    }
    while (Q--) {
        int l, r, x;
        cin >> l >> r >> x;
        l--; r--;
        int mx = x, cnt = (r - l + 1) % 2, idx = r;
        if (l > 0) { // has prefix
            Node prefix = seg.query(0, l - 1);
            if (prefix.mx == mx) {
                cnt ^= prefix.cnt;
            } else if (prefix.mx > mx && prefix.cnt) {
                mx = prefix.mx;
                cnt = prefix.cnt;
                idx = prefix.idx;
            }
        }
        if (r < N - 1) { // has suffix
            Node suffix = seg.query(r + 1, N - 1);
            if (suffix.mx == mx) {
                cnt ^= suffix.cnt;
                idx = suffix.idx;
            } else if (suffix.mx > mx || !cnt) {
                mx = suffix.mx;
                cnt = suffix.cnt;
                idx = suffix.idx;
            }
        }
        if (cnt) {
            cout << idx + 1 << endl;
        } else {
            cout << N + 1 << endl;
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
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
int p, x, y, n;
const int MOD = 998244353, MAXN = 1e5 + 5;
vector<int> P, Q;

int inv(int i) {
  return i <= 1 ? i : MOD - (int)(MOD/i) * inv(MOD % i) % MOD;
}

vector<int> fact, inv_fact;

void factorials(int n) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % MOD;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1]);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD;
    }
}

// combinations selecting r from n items
int choose(int n, int r) {
    if (n < r) return 0;
    return (fact[n] * inv_fact[r] % MOD) * inv_fact[n - r] % MOD;
}

// m successes from n trials
int binomial_dist(int n, int m) {
    return (choose(n, m) * P[m]) % MOD * Q[n - m] % MOD;
}

void solve() {
    cin >> p >> x >> y >> n;
    P.resize(n + 1); Q.resize(n + 1);
    int p_inv = p * inv(100) % MOD;
    int q_inv = (100 - p) * inv(100) % MOD;
    P[0] = 1; Q[0] = 1;
    for (int i = 1; i <= n; i++) {
        P[i] = P[i - 1] * p_inv % MOD;
        Q[i] = Q[i - 1] * q_inv % MOD;
    }
    int ans = 0;
    for (int a = 0; a <= n; a++) {
        int trade = (n - a) / x;
        int val = a + trade * y;
        ans = (ans + (binomial_dist(n, a) * val) % MOD) % MOD;
    }
    cout << ans << endl;
}
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    factorials(MAXN);
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