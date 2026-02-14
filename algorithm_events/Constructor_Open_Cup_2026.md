# Constructor Open Cup 2026

# Practice Round

## E. Chocolate Split

### Solution 1: finite arithmetic series, math, formula

```cpp
int K;

int64 calc(int64 n) {
    return n * (n + 1) / 2;
}

void solve() {
    cin >> K;
    K += 2;
    int64 ans = calc(K / 2) + calc((K - 1) / 2);
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

## F. Divisibility Problem

### Solution 1: dfs, backtracking, recursion

```cpp
int N, M, ans;

bool dfs(int i, int rem, int cand) {
    if (i == N) {
        if (rem == 0) ans = cand;
        return rem == 0;
    }
    for (int d = 1; d <= 2; ++d) {
        if (dfs(i + 1, (rem * 10 + d) % M, cand * 10 + d)) return true;
    }
    return false;
}

void solve() {
    cin >> N;
    M = 1;
    for (int i = 0; i < N; ++i) {
        M <<= 1;
    }
    dfs(0, 0, 0);
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

## G. Comics Collection

### Solution 1: principle of inclusion-exclusion

This is a good introduction to the principle of inclusion-exclusion. We can count how many numbers from 1 to N are divisible by 5, 3, and 2, and then use inclusion-exclusion to find how many numbers are divisible by at least one of them. Finally, we can calculate the sum of all these numbers.

```cpp
int N;

void solve() {
    cin >> N;
    int countFive = N / 5;
    int countThree = N / 3 - N / 15;
    int countTwo = N / 2 - N / 6 - N / 10 + N / 30;
    int countOne = N - countTwo - countThree - countFive;
    int64 ans = 5LL * countFive + 3LL * countThree + 2LL * countTwo + 1LL * countOne;
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

## H. Bad Sectors

### Solution 1: scanning and tracking distance from last bad sector, reverse, symmetry

```cpp
int N, K;
string S, ans;

void update() {
    for (int i = 0, d = K + 1; i < N; ++i, ++d) {
        if (S[i] == '*') d = 0;
        if (d <= K) ans[i] = '*';
    }
}

void solve() {
    cin >> N >> K >> S;
    ans.assign(N, '.');
    update();
    reverse(S.begin(), S.end());
    reverse(ans.begin(), ans.end());
    update();
    reverse(ans.begin(), ans.end());
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

## I. Friends at the Cafeteria

### Solution 1: line sweep, sorting, two pointers, fixed sized window, counting active intervals

I think easiest way is to collect all the start and end times, sort them, and then do a line sweep.  Where you define the right endpoint to be that element, and you can calculate the left endpoint, and determine how many intervals currently overlap with it. 

```cpp
int N, M;
vector<int> A, B;

void solve() {
    cin >> N >> M;
    A.assign(N, 0);
    B.assign(N, 0);
    vector<int> events;
    for (int i = 0; i < N; i++) {
        cin >> A[i] >> B[i];
        B[i] += A[i];
        events.emplace_back(A[i]);
        events.emplace_back(B[i]);
    }
    sort(A.begin(), A.end());
    sort(B.begin(), B.end());
    sort(events.begin(), events.end());
    events.erase(unique(events.begin(), events.end()), events.end());
    int ans = 0, cnt = 0, i = 0, j = 0;
    for (int r : events) {
        int l = r - M;
        while (i < N && A[i] <= r) {
            cnt++;
            i++;
        }
        while (j < N && B[j] < l) {
            cnt--;
            j++;
        }
        ans = max(ans, cnt);
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


## J. Math Exam

### Solution 1: greedy feasibility check, binary search

It is feasible if I can use all elements in the array to create sums that are less than or equal to the target.  Basically for the values we will have feasibility looking like FFFTTTT, and you want to return the smallest T. 

```cpp
int N, K;
vector<int> A;

bool possible(int64 target) {
    for (int k = 0, i = 0; k < K; ++k) {
        int64 sum = 0;
        while (i < N && sum + A[i] <= target) {
            sum += A[i++];
        }
        if (i == N) return true;
    }
    return false;
}

void solve() {
    cin >> N >> K;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    int64 lo = 0, hi = 1e16;
    while (lo < hi) {
        int64 mid = lo + (hi - lo) / 2;
        if (possible(mid)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    cout << lo << endl;
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
## K. The Alarm

### Solution 1: 

I was being dumb it was just SCC condensation and counting number of nodes with indegree equal to 0, because only these can reach all nodes downstream from it. 

```cpp

```

## L. Extended Fibonacci

### Solution 1: 

Finishing it up, it is linear diophantine equation. 

```cpp

```

# Final Round

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

