# Linear Algebra

## Rank of Matrix

### Solution 1: gaussian elimination, row echelon form, pivots

Rank is just the number of pivots, or number of linearly independent rows, which is the number of pivots. 

```cpp
const int64 MOD = 998244353;
int N, M;
vector<vector<int64>> mat;

int64 inv(int i, int64 m) {
    return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

void solve() {
    cin >> N >> M;
    mat.assign(N, vector<int64>(M + 1, 0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            cin >> mat[i][j];
        }
    }
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
        for (int i = c; i < M; ++i) {
            mat[r][i] = pinv * mat[r][i] % MOD;
        }
        for (int i = r + 1; i < N; ++i) {
            if (!mat[i][c]) continue;
            int64 factor = mat[i][c];
            for (int j = c; j < M; ++j) {
                int64 term = factor * mat[r][j] % MOD;
                mat[i][j] = (mat[i][j] - term + MOD) % MOD;
            }
        }
        pivots.emplace_back(c);
        ++r;
    }
    int ans = pivots.size();
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

## Rank of Matrix (Mod 2)

### Solution 1:  guassian elimination, row echelon form, pivots, bit words

Bit packing into 64 bit integers, so now you can perform 64 bitwise operations in parallel.

```cpp
const int BITS = 64;
int N, M;
vector<vector<uint64>> mat;

int ceil(int x, int y) {
    return (x + y - 1) / y;
}

void solve() {
    cin >> N >> M;
    int W = ceil(M, BITS);
    mat.assign(N, vector<uint64>(W, 0));
    for (int i = 0; i < N; ++i) {
        string s;
        cin >> s;
        for (int j = 0; j < M; ++j) {
            int v = s[j] - '0';
            if (!v) continue;
            mat[i][j >> 6] |= (1ULL << (j % BITS));
        }
    }
    vector<int> pivots;
    for (int c = 0, r = 0; c < M && r < N; ++c) {
        int blk = c >> 6, sel = -1;
        uint64 mask = (1ULL << (c % BITS));
        for (int i = r; i < N; ++i) {
            if (mat[i][blk] & mask) {
                sel = i;
                break;
            }
        }
        if (sel == -1) continue;
        swap(mat[sel], mat[r]);
        for (int i = r + 1; i < N; ++i) {
            if (!(mat[i][blk] & mask)) continue;
            for (int j = blk; j < W; ++j) {
                mat[i][j] ^= mat[r][j];
            }
        }
        pivots.emplace_back(c);
        ++r;
    }
    int ans = pivots.size();
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

## System of Linear Equations

### Solution 1: gaussian elimination, row echelon form, pivots, free variables, rank

I still don't know what it means by rank of solution, it seems to be the size of the nullspace, that is in reference to the free variables. 
So if I have 3 variables, and 2 pivot columns, that means 1 variable is free, and also that is the rank. 
Then it asks to find the basis of the solution

Given $Ax=b$

So when it asks for rank of solution, it does not mean the rank of the matrix which would be the number of linearly independent rows. 
But it means what is often times called nullity of a matrix.  This refers to the dimension of its null space (or kernel).  The null space is the set of all vectors that, when multiplied by the matrix, result in the zero vector.  Essentially, nullity tells you how many degrees of freedom exist in the solutions to the homogenous system represented by the matrix. 

Degrees of Freedom:
You start with n variable with free choicese before imposing the equations. 
Treat each equation as a constraint, and each independent equations removes 1 dof.
The leftover free choices is the number of degrees of freedom. 

if dof = 0, there is a unique solution if consistent, else no solution
if dof > 0, there are infinitely many solutions, parameterized by the free variables. 

The rank-nullity theorem comes in clutch here, where the degrees of freedom = n - rank(A)

```cpp
const int64 MOD = 998244353;
int N, M; // rows, columns
vector<vector<int64>> mat;

int64 inv(int i, int64 m) {
    return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

void solve() {
    cin >> N >> M;
    mat.assign(N, vector<int64>(M + 1, 0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            cin >> mat[i][j];
        }
    }
    for (int i = 0; i < N; ++i) {
        cin >> mat[i][M];
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
    vector<int> free;
    for (int i = 0, j = 0; i < M; ++i) {
        if (j < numPivots && pivots[j] == i) {
            ++j;
            continue;
        }
        free.emplace_back(i);
    }
    // basis? what is this actually? 
    int rank = M - numPivots;
    vector<vector<int64>> basis(rank, vector<int64>(M, 0));
    for (int i = 0; i < rank; ++i) {
        // set the free var to 1
        basis[i][free[i]] = 1;
        for (int j = numPivots - 1; j >= 0; --j) {
            int c = pivots[j];
            int64 sum = 0;
            for (int k = c + 1; k < M; ++k) {
                sum = (sum + mat[j][k] * basis[i][k] % MOD) % MOD;
            }
            basis[i][c] = (basis[i][c] - sum + MOD) % MOD;
        }
    }
    cout << rank << endl;
    for (int x : X) {
        cout << x << " ";
    }
    cout << endl;
    for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < M; ++j) {
            cout << basis[i][j] << " ";
        }
        cout << endl;
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

## System of Linear Equations (Mod 2)

### Solution 1: 

```cpp

```