# Atcoder Beginner Contest 356

## 

### Solution 1: 

```cpp
void solve() {
    int N;
    cin >> N;
    vector<int> A(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    vector<int> lmax(N), rmax(N);
    stack<int> stk;
    for (int i = 0; i < N; i++) {
        while (!stk.empty() && A[i] >= A[stk.top()]) {
            stk.pop();
        }
        lmax[i] = i - (stk.empty() ? -1 : stk.top());
        stk.push(i);
    }
    while (!stk.empty()) {
        stk.pop();
    }
    for (int i = N - 1; i >= 0; i--) {
        while (!stk.empty() && A[i] > A[stk.top()]) {
            stk.pop();
        }
        rmax[i] = (stk.empty() ? N : stk.top()) - i;
        stk.push(i);
    }
    vector<int> lmin(N), rmin(N);
    while (!stk.empty()) {
        stk.pop();
    }
    for (int i = 0; i < N; i++) {
        while (!stk.empty() && A[i] <= A[stk.top()]) stk.pop();
        lmin[i] = i - (stk.empty() ? -1 : stk.top());
        stk.push(i);
    }
    while (!stk.empty()) {
        stk.pop();
    }
    for (int i = N - 1; i >= 0; i--) {
        while (!stk.empty() && A[i] < A[stk.top()]) stk.pop();
        rmin[i] = (stk.empty() ? N : stk.top()) - i;
        stk.push(i);
    }
    int ans = 0;
    for (int i = 0; i < N; i++) {
        int lcount = min(lmax[i], lmin[i]), rcount = min(rmax[i], rmin[i]);
        ans += A[i] * lcount * rcount;
    }
    cout << ans << endl;
}

signed main() {
    solve();
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