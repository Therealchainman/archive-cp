# Helvetic Coding Contest 2024

## 

### Solution 1:  

```py

```

## B3. Exact Neighbours (Hard)

### Solution 1:  

```py

```

## C3. Game on Tree (Hard)

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```

## E3. Trails (Medium)

### Solution 1:  dynamic programming, matrix exponentiation

```cpp
const int MOD = 1e9 + 7;
vector<vector<int>> transition_matrix, base_matrix;

vector<vector<int>> mat_mul(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2) {
    int rows1 = mat1.size(), cols1 = mat1[0].size();
    int rows2 = mat2.size(), cols2 = mat2[0].size();
    vector<vector<int>> result_matrix(rows1, vector<int>(cols2, 0));
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            for (int k = 0; k < cols1; k++) {
                result_matrix[i][j] = (result_matrix[i][j] + (long long)mat1[i][k] * mat2[k][j]) % MOD;
            }
        }
    }
    return result_matrix;
}

vector<vector<int>> mat_pow(const vector<vector<int>>& matrix, int power) {
    if (power <= 0) {
        cout << "n must be non-negative integer" << endl;
        return {};
    }
    if (power == 1) return matrix;
    if (power == 2) return mat_mul(matrix, matrix);

    vector<vector<int>> t1 = mat_pow(matrix, power / 2);
    if (power % 2 == 0) {
        return mat_mul(t1, t1);
    }
    return mat_mul(t1, mat_mul(matrix, t1));
}

void solve() {
    int m, n;
    cin >> m >> n;
    vector<int> s(m), l(m);
    for (int i = 0; i < m; i++) cin >> s[i];
    for (int i = 0; i < m; i++) cin >> l[i];

    transition_matrix.assign(m, vector<int>(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            transition_matrix[i][j] = s[i] * s[j] + s[i] * l[j] + l[i] * s[j];
        }
    }

    base_matrix.assign(m, vector<int>(1, 0));
    base_matrix[0][0] = 1;
    vector<vector<int>> exponentiated_matrix = mat_pow(transition_matrix, n);
    vector<vector<int>> solution_matrix = mat_mul(exponentiated_matrix, base_matrix);

    int ans = 0;
    for (int i = 0; i < m; i++) {
        ans = (ans + solution_matrix[i][0]) % MOD;
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

## E3. Trails (Hard)

### Solution 1:  

```py

```

##

### Solution 1:  

```py

```

##

### Solution 1:  

```py

```