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

### Solution 1:  partially ordered sets, reachability in DAG, Dilworth's theorem, stacks, greedy, directed acyclic graph

```cpp
int N;
string resp;

bool ask(int i, int j) {
    cout << "? " << i << " " << j << endl;
    cout.flush();
    cin >> resp;
    return resp == "YES";
}

void solve() {
    cin >> N;
    vector<int> ans(N + 1, 0);
    vector<int> white, black, mixed;
    for (int i = 1; i <= N; i++) {
        bool white_reachable = false, black_reachable = false, mixed_reachable = false;
        if (mixed.empty()) {
            if (!white.empty()) {
                if (ask(white.end()[-1], i)) white_reachable = true;
            }
            if (!black.empty()) {
                if (ask(black.end()[-1], i)) black_reachable = true;
                if (resp == "YES") black_reachable = true;
            }
            if (white_reachable && black_reachable) {
                mixed.push_back(i);
            } else if (white_reachable) {
                white.push_back(i);
            } else if (black_reachable) {
                black.push_back(i);
            } else if (white.empty()) {
                white.push_back(i);
            } else {
                black.push_back(i);
            }
        } else {
            if (ask(mixed.end()[-1], i)) mixed_reachable = true;
            if (ask(white.end()[-1], i)) white_reachable = true;
            if (mixed_reachable) {
                mixed.push_back(i);
            } else {  
                if (white_reachable) white.push_back(i);
                else black.push_back(i);
                for (int j : mixed) {
                    if (white_reachable) black.push_back(j);
                    else white.push_back(j);
                }
                mixed.clear();
            }
        }
    }
    for (int v : white) {
        ans[v] = 1;
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