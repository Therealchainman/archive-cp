# Bitmask DP


## Decision problem that uses patterns in binary encoding

- Uses matching prefix of binary encoding to merge constraints and half them at each step
- Slowly discovers the set S, so at each step only part of it is known like S = xx10,  still x are undecided if they wil lbe 0 or 1. 

Also look at the pattern of binary encoding that is there is a pair where the prefix if you remove the last bit or decide on it, the rest is the same.
000
001
010
011

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
    for (int T = 0; T < 1 << (N - depth); T += 2) {
        new_constraints[T >> 1] = constraints[T] & constraints[T + 1];
    }
    dfs(S, depth + 1, new_constraints);
    // S WILL CONTAIN ELEMENT EQUIVALENT TO VALUE OF DEPTH
    for (int T = 0; T < 1 << (N - depth); T += 2) {
        new_constraints[T >> 1] = constraints[T] & (constraints[T + 1] >> 1);
    }
    dfs(S | (1 << depth), depth + 1, new_constraints);
}
 
void solve() {
    cin >> N;
    int num_sets = 1 << N;
    vector<int> constraints(num_sets);
    constraints[0] = (1 << (N + 1)) - 1;
    // check this
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
```