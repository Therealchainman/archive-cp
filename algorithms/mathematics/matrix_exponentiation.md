# Matrix Exponentiation

## Implementation in python

Can be used to solve a system of linear equations with matrix.

AX = B for example, where AX represents matrix multiplication of A and X.

```py
"""
matrix multiplication with modulus
"""
def mat_mul(mat1, mat2, mod):
    result_matrix = []
    for i in range(len(mat1)):
        result_matrix.append([0]*len(mat2[0]))
        for j in range(len(mat2[0])):
            for k in range(len(mat1[0])):
                result_matrix[i][j] += (mat1[i][k]*mat2[k][j])%mod
    return result_matrix

"""
matrix exponentiation with modulus
matrix is represented as list of lists in python
"""
def mat_pow(matrix, power, mod):
    if power<=0:
        print('n must be non-negative integer')
        return None
    if power==1:
        return matrix
    if power==2:
        return mat_mul(matrix, matrix, mod)
    t1 = mat_pow(matrix, power//2, mod)
    if power%2 == 0:
        return mat_mul(t1, t1, mod)
    return mat_mul(t1, mat_mul(matrix, t1, mod), mod)
```

Often times you are solving with a transition matrix, which means it allows you to compute the a_n+1 term from the a_n term.

transition_matrix^power*base_matrix = solution_matrix

How can this be applied is from this example. 

This solves a sum of geometrix progression type problem where you want
sum = base^0 + base^1 + base^2 + ... + base^(num_terms-1)

```py
base, num_terms, mod = 3, 4, 7
# exponentiated_matrix*base_matrix = solution_matrix
# exponentiated_matrix = transition_matrix^num_terms
transition_matrix = [[base, 1], [0, 1]]
base_matrix = [[0], [1]]
exponentiated_matrix = mat_pow(transition_matrix, num_terms, mod)
solution_matrix = mat_mul(exponentiated_matrix, base_matrix, mod)
return solution_matrix[0][0]
```

## Implementation in C++

Example, just fill in transition matrix and base matrix to solve the specific task.

```cpp
const int MOD = 1e9 + 7;
vector<vector<int64>> transitionMatrix, baseMatrix;

vector<vector<int64>> matMul(const vector<vector<int64>>& mat1, const vector<vector<int64>>& mat2) {
    int rows1 = mat1.size(), cols1 = mat1[0].size();
    int rows2 = mat2.size(), cols2 = mat2[0].size();
    vector<vector<int64>> resultMatrix(rows1, vector<int64>(cols2, 0));
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            for (int k = 0; k < cols1; k++) {
                resultMatrix[i][j] = (resultMatrix[i][j] + mat1[i][k] * mat2[k][j]) % MOD;
            }
        }
    }
    return resultMatrix;
}

vector<vector<int64>> matPow(const vector<vector<int64>>& matrix, int power) {
    if (power <= 0) {
        cout << "n must be non-negative integer" << endl;
        return {};
    }
    if (power == 1) return matrix;
    if (power == 2) return matMul(matrix, matrix);

    vector<vector<int64>> t1 = matPow(matrix, power / 2);
    if (power % 2 == 0) {
        return matMul(t1, t1);
    }
    return matMul(t1, matMul(matrix, t1));
}

// this example is right in the idea
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
```