# Codeforces Round 948 Div 2

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## E. Tensor

### Solution 1: 

```cpp
int N;
vector<int> ans;
string resp;

void solve() {
    cin >> N;
    ans.assign(N + 1, -1);
    vector<int> U(2, -1), V(2, -1);
    V[0] = N;
    U[0] = V[0] - 1;
    ans[N] = 0;
    bool turn = true;
    while (U[0] > 0 || U[1] > 0) {
        int idx;
        if (U[0] > U[1] || (U[0] == U[1] && turn)) {
            idx = 0;
        } else {
            idx = 1;
        }
        cout << "?" << " " << U[idx] << " " << V[idx] << endl;
        cout.flush();
        cin >> resp;
        if (resp == "YES" && ans[U[idx]] == -1) {
            if (U[idx] == U[idx ^ 1]) turn = false;
            V[idx] = U[idx];
            ans[V[idx]] = idx;
        } else if (V[idx ^ 1] == -1) {
            V[idx ^ 1] = U[idx];
            ans[V[idx ^ 1]] = idx ^ 1;
            U[idx ^ 1] = V[idx ^ 1] - 1;
        }
        U[idx]--;
    }
    cout << "! ";
    for (int i = 1; i <= N; i++) {
        cout << ans[i] << " ";
    }
    cout << endl;
    cout.flush();
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