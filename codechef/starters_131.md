# Starters 131

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## Chef needs to return some videotapes

### Solution 1:  square root decomposition, sorted list (set), binary search, prefix sum per block

```cpp
const int MAXN = 1e5 + 5, B = 450;
int N, Q, last, t, a, b, lo, hi;
int C[MAXN], L[MAXN];
vector<int> psum, prv;
vector<int> block_order;
vector<set<int>> indices;

void recalc(int bid) {
    lo = bid * B, hi = min(N, (bid + 1) * B);
    sort(block_order.begin() + lo, block_order.begin() + hi, [&](int x, int y) {
        return prv[x] < prv[y];
    });
    for (int i = lo; i < hi; i++) {
        psum[i] = L[block_order[i]];
        if (i > lo) psum[i] += psum[i - 1];
    }
}

void solve() {
    cin >> N >> Q;
    indices.assign(N + 1, set<int>());
    for (int i = 0; i < N; i++) {
        cin >> L[i] >> C[i];
        C[i]--;
        indices[C[i]].insert(i);
    }
    prv.assign(N + 1, -1);
    last = 0;
    psum.assign(N, 0);
    block_order.resize(N);
    iota(block_order.begin(), block_order.end(), 0);
    vector<int> seen(N, -1);
    for (int i = 0; i < N; i++) {
        prv[i] = seen[C[i]];
        seen[C[i]] = i;
    }
    for (int i = 0; i < N; i += B) {
        recalc(i / B);
    }
    for (int q = 0; q < Q; q++) {
        cin >> t >> a >> b;
        if (t == 1) {
            int l, r;
            l = a ^ last;
            r = b ^ last;
            l--; r--;
            int lb = l / B, rb = r / B;
            last = 0;
            for (int bid = lb + 1; bid < rb; bid++) {
                lo = bid * B, hi = min(N, (bid + 1) * B);
                int idx = lower_bound(block_order.begin() + lo, block_order.begin() + hi, l, [&](int x, int y) {
                    return prv[x] < y;
                }) - block_order.begin();
                if (idx > lo) last += psum[idx - 1];
            }
            for (int i = l; i < min(r + 1, (lb + 1) * B); i++) {
                if (prv[i] < l) last += L[i];
            }
            if (lb != rb) {
                for (int i = rb * B; i <= r; i++) {
                    if (prv[i] < l) last += L[i];
                }
            }
            cout << last << endl;
        } else if (t == 2) {
            int x, y;
            x = a ^ last;
            y = b ^ last;
            x--;
            L[x] = y;
            recalc(x / B);
        } else {
            int i, y, j, k;
            i = a ^ last;
            y = b ^ last;
            i--; 
            y--;
            j = N; k = N;
            auto it = indices[C[i]].find(i);
            if (next(it) != indices[C[i]].end()) {
                k = *next(it);
                prv[k] = prv[i];
            }
            indices[C[i]].erase(i);
            C[i] = y;
            indices[C[i]].insert(i);
            auto it2 = indices[y].find(i);
            if (next(it2) != indices[y].end()) {
                j = *next(it2);
                prv[j] = i;
            }
            if (it2 == indices[C[i]].begin()) {
                prv[i] = -1;
            } else {
                prv[i] = *prev(it2);
            }
            recalc(i / B);
            if (j < N) recalc(j / B);
            if (k < N) recalc(k / B);
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    while (T--) {
        solve();
    }
    return 0;
}
```