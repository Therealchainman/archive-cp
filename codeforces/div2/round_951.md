# Codeforces Round 951 Div 2

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: Hamiltonian Path, undirected graph, recursion, deque

```cpp
deque<int> path;

pair<int, int> ask(int d) {
    cout << "? " << d << endl;
    cout.flush();
    int u, v;
    cin >> u >> v;
    return {u, v};
}

void recurse(int n) {
    if (n == 1) {
        auto [u, _] = ask(0);
        path.push_front(u);
        return;
    }
    if (n == 2) {
        auto [u, b1] = ask(0);
        auto [v, b2] = ask(0);
        path.push_front(u);
        path.push_front(v);
        return;
    }
    auto [u, b] = ask(n - 2);
    if (b == 0) {
        auto [v, _] = ask(0);
        recurse(n - 2);
        path.push_front(u);
        path.push_front(v);
    } else {
        recurse(n - 1);
        if (path.front() == b) {
            path.push_back(u);
        } else {
            path.push_front(u);
        }
    }
}

void solve() {
    int n;
    cin >> n;
    path.clear();
    recurse(n);
    cout << "! ";
    for (int x : path) {
        cout << x << " ";
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