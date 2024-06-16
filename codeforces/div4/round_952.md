# Codeforces Round 952 Div 4

## H1. Maximize the Largest Component (Easy Version)

### Solution 1:  matrix, grid, dfs, connected components, set

```cpp
int R, C, comp, ans, csize;
vector<vector<int>> mat, comps;
vector<int> sizes;
set<int> vis;

// 0 is .
// 1 is #
bool in_bounds(int r, int c) {
    return r >= 0 && r < R && c >= 0 && c < C;
}

void dfs(int r, int c) {
    if (!in_bounds(r, c) || !mat[r][c]) return;
    if (mat[r][c] == 2) return;
    mat[r][c] = 2;
    csize++;
    comps[r][c] = comp;
    dfs(r + 1, c);
    dfs(r - 1, c);
    dfs(r, c + 1);
    dfs(r, c - 1);
}
void solve() {
    cin >> R >> C;
    mat.assign(R, vector<int>(C, 0));
    comps.assign(R, vector<int>(C, 0));
    for (int r = 0; r < R; r++) {
        string row;
        cin >> row;
        for (int c = 0; c < C; c++) {
            mat[r][c] = row[c] == '.' ? 0 : 1;
        }
    }
    ans = 0, comp = 1, csize = 0;
    sizes.clear();
    sizes.push_back(0);
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            if (mat[r][c] == 1 && !comps[r][c]) {
                csize = 0;
                dfs(r, c);
                sizes.push_back(csize);
                comp++;
            }
        }
    }
    // row queries
    for (int r = 0; r < R; r++) {
        vis.clear();
        int sz = 0;
        for (int c = 0; c < C; c++) {
            if (r > 0 && comps[r - 1][c] && vis.find(comps[r - 1][c]) == vis.end()) {
                sz += sizes[comps[r - 1][c]];
                vis.insert(comps[r - 1][c]);
            }
            if (r + 1 < R && comps[r + 1][c] && vis.find(comps[r + 1][c]) == vis.end()) {
                sz += sizes[comps[r + 1][c]];
                vis.insert(comps[r + 1][c]);
            }
            if (comps[r][c] && vis.find(comps[r][c]) == vis.end()) {
                sz += sizes[comps[r][c]];
                vis.insert(comps[r][c]);
            }
            if (!comps[r][c]) sz++;
        }
        ans = max(ans, sz);
    }
    // col queries
    for (int c = 0; c < C; c++) {
        vis.clear();
        int sz = 0;
        for (int r = 0; r < R; r++) {
            if (c > 0 && comps[r][c - 1] && vis.find(comps[r][c - 1]) == vis.end()) {
                sz += sizes[comps[r][c - 1]];
                vis.insert(comps[r][c - 1]);
            }
            if (c + 1 < C && comps[r][c + 1] && vis.find(comps[r][c + 1]) == vis.end()) {
                sz += sizes[comps[r][c + 1]];
                vis.insert(comps[r][c + 1]);
            }
            if (comps[r][c] && vis.find(comps[r][c]) == vis.end()) {
                sz += sizes[comps[r][c]];
                vis.insert(comps[r][c]);
            }
            if (!comps[r][c]) sz++;
        }
        ans = max(ans, sz);
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

## H2. Maximize the Largest Component (Hard Version)

### Solution 1: 

```cpp

```