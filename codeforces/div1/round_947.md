# Codeforces Round 947 div1

## Paint Tree

### Solution 1: 

```py

```

## E. Chain Queries

### Solution 1: 

```py

```

## F. Set

### Solution 1:  set theory, bitmasks, binary encoding, recursion, decision tree, merging constraints

```cpp
int N;
vector<int> ans;

void dfs(int S, int depth, vector<int> constraints) {
    if (depth == N) {
        if (constraints[0] & 1) ans.push_back(S);
        return;
    }
    // S WILL NOT CONTAIN ELEMENT EQUIVALENT TO VALUE OF DEPTH
    vector<int> new_constraints(1 << (N - depth - 1));
    int M = 1 << (N - depth);
    for (int T = 0; T < M; T += 2) {
        new_constraints[T >> 1] = constraints[T] & constraints[T + 1];
    }
    dfs(S, depth + 1, new_constraints);
    // S WILL CONTAIN ELEMENT EQUIVALENT TO VALUE OF DEPTH
    for (int T = 0; T < M; T += 2) {
        new_constraints[T >> 1] = constraints[T] & (constraints[T + 1] >> 1);
    }
    dfs(S | (1 << depth), depth + 1, new_constraints);
}

void solve() {
    cin >> N;
    int num_sets = 1 << N;
    vector<int> constraints(num_sets);
    constraints[0] = (1 << (N + 1)) - 1;
    for (int T = 1; T < num_sets; T++) {
        cin >> constraints[T];
    }
    dfs(0, 0, constraints);
    cout << ans.size() << endl;
    sort(ans.begin(), ans.end());
    for (int x : ans) {
        cout << x << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## G. Zimpha Fan Club

### Solution 1: 

```py

```

## H. 378QAQ and Core

### Solution 1: 

```py

```

## I. Mind Bloom

### Solution 1: 

```py

```