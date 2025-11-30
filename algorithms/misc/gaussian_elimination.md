# Gaussian Elimination


## Gaussian Elimination used to solve system of linear equations

Use forward elimination algorithm and back substitution to find the solution fo the system of linear equations.  The forward elimination transforms the matrix into row echelon form. 

```cpp
const int64 MOD = 1e9 + 7;
int N, M; // rows, columns
vector<vector<int64>> mat;

int64 inv(int i, int64 m) {
    return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

void solve() {
    cin >> N >> M;
    mat.assign(N, vector<int64>(M + 1, 0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= M; ++j) {
            cin >> mat[i][j];
        }
    }
    // forward elimination
    vector<int> pivots;
    for (int c = 0, r = 0; c < M && r < N; ++c) {
        int selectedPivot = -1;
        for (int i = r; i < N; ++i) {
            if (mat[i][c]) {
                selectedPivot = i;
                break;
            }
        }
        if (selectedPivot == -1) continue;
        swap(mat[selectedPivot], mat[r]); // swap
        int64 pinv = inv(mat[r][c], MOD); // normalize pivot to 1
        for (int i = c; i <= M; ++i) {
            mat[r][i] = pinv * mat[r][i] % MOD;
        }
        for (int i = r + 1; i < N; ++i) {
            if (!mat[i][c]) continue;
            int64 factor = mat[i][c];
            for (int j = c; j <= M; ++j) {
                int64 term = factor * mat[r][j] % MOD;
                mat[i][j] = (mat[i][j] - term + MOD) % MOD;
            }
        }
        pivots.emplace_back(c);
        ++r;
    }
    int numPivots = pivots.size();
    // check for any contradiction
    for (int i = numPivots; i < N; ++i) {
        if (mat[i][M]) {
            cout << -1 << endl;
            return;
        }
    }
    // back substition
    vector<int> X(M, 0);
    for (int i = numPivots - 1; i >= 0; --i) {
        int c = pivots[i];
        int64 x = mat[i][M];
        for (int j = c + 1; j < M; ++j) {
            int64 term = mat[i][j] * X[j] % MOD;
            x = (x - term + MOD) % MOD;
        }
        X[c] = x;
    }
    for (int x : X) {
        cout << x << " ";
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```