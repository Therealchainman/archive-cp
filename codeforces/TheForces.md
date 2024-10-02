# TheForces Round 25

## E. Range Modulo Queries

### sparse tables, gcd range query, max range query, binary search, precompute factors for all integers in range

This TLE but probably because it is in python

```py
import math
import bisect
LIM = int(1e6)
LOG = 21
factors = [[] for _ in range(LIM + 1)]

def main():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    st_gcd = [[0] * n for _ in range(LOG)]
    st_gcd[0] = [x - y for x, y in zip(a, b)]
    st_max = [[-math.inf] * n for _ in range(LOG)]
    st_max[0] = b[:]
    for i in range(1, LOG):
        j = 0
        while (j + (1 << (i - 1))) < n:
            st_gcd[i][j] = math.gcd(st_gcd[i - 1][j], st_gcd[i - 1][j + (1 << (i - 1))])
            st_max[i][j] = max(st_max[i - 1][j], st_max[i - 1][j + (1 << (i - 1))])
            j += 1
    def query(left, right):
        length = right - left + 1
        k = int(math.log2(length))
        return math.gcd(st_gcd[k][left], st_gcd[k][right - (1 << k) + 1]), max(st_max[k][left], st_max[k][right - (1 << k) + 1])
    for _ in range(q):
        left, right = map(int, input().split())
        left -= 1
        right -= 1
        g, m = query(left, right)
        if g <= m: 
            print(-1)
            continue
        i = bisect.bisect_right(factors[g], m)
        res = factors[g][i]
        print(res)

if __name__ == '__main__':
    T = int(input())
    for i in range(1, LIM + 1):
        for j in range(i, LIM + 1, i):
            factors[j].append(i)
    for _ in range(T):
        main()
```

```cpp
const int LIM = 1e6, LOG = 21;
vector<vector<int>> factors(LIM + 1);
int A[LIM + 1], B[LIM + 1];
int st_gcd[21][LIM + 1], st_max[21][LIM + 1];

int query_gcd(int L, int R) {
    int k = log2(R - L + 1);
    return gcd(st_gcd[k][L], st_gcd[k][R - (1LL << k) + 1]);
}

int query_max(int L, int R) {
    int k = log2(R - L + 1);
    return max(st_max[k][L], st_max[k][R - (1LL << k) + 1]);
}

void solve() {
    int N, Q;
    cin >> N >> Q;
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
    }
    for (int i = 0; i < N; i++) {
        st_gcd[0][i] = A[i] - B[i] > 0 ? A[i] - B[i] : 1;
        st_max[0][i] = B[i];
    }
    for (int i = 1; i < LOG; i++) {
        for (int j = 0; j + (1LL << (i - 1)) < N; j++) {
            st_gcd[i][j] = gcd(st_gcd[i - 1][j], st_gcd[i - 1][j + (1LL << (i - 1))]);
            st_max[i][j] = max(st_max[i - 1][j], st_max[i - 1][j + (1LL << (i - 1))]);
        }
    }
    while (Q--) {
        int L, R;
        cin >> L >> R;
        L--;
        R--;
        int g = query_gcd(L, R);
        int m = query_max(L, R);
        if (g <= m) {
            printf("-1\n");
        } else {
            int res = *upper_bound(factors[g].begin(), factors[g].end(), m);
            printf("%d\n", res);
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    for (int i = 1; i <= LIM; i++) {
        for (int j = i; j <= LIM; j += i) {
            factors[j].push_back(i);
        }
    }
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```
# TheForces Round 34

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

# TheForces Round 35

## A. Simple Update - I
 
### Solution 1:  sliding window, counting 0s and 1s

```cpp
int N, K;
string S;
 
void solve() {
    cin >> N >> K >> S;
    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += S[i] - '0';
    }
    int cur = sum, ans = sum;
    // window
    for (int i = 0; i < 2 * K; i++) {
        if (i < K) {
            if (S[i] == '0') cur++;
        } else {
            if (S[i] == '1') cur--;
        }
    }
    ans = max(ans, cur);
    for (int i = 2 * K; i < N; i++) {
        cur++;
        if (S[i] == '1') cur--;
        ans = max(ans, cur);
    }
    cout << ans << endl;
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

## B. Simple Update - II

### Solution 1:  first and last 0, ceil

```cpp
int N, K;
string S;
 
int ceil(int x, int y) {
    return (x + y - 1) / y;
}
 
void solve() {
    cin >> N >> K >> S;
    int zeros = 0;
    for (int i = 0; i < N; i++) {
        if (S[i] == '0') zeros++;
    }
    if (zeros <= K) {
        cout << 0 << endl;
        return;
    }
    int l = 0, r = 0;
    for (int i = 0; i < N; i++) {
        if (S[i] == '0') r = i + 1;
    }
    r -= K;
    for (int i = N - 1; i >= 0; i--) {
        if (S[i] == '0') l = i;
    }
    int ans = ceil(r - l, K);
    cout << ans << endl;
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

## C. Yet Another Nim Game

### Solution 1:  parity, loop

```cpp
int N;
vector<int> A;
 
void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int ans = 0, last = 0;
    for (int i = 1; i <= N; i++) {
        if (A[i - 1] & 1) last = i;
        ans += last;
    }
    cout << ans << endl;
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

## D. String From Another World

### Solution 1:  dynamic programming

```cpp

```

# TheForces Round 36

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

