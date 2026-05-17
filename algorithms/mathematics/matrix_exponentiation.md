# Matrix Exponentiation

Repeatedly multiplying a square matrix by itself.

## Implementation in python

Can be used to solve a system of linear equations with matrix.

AX = B for example, where AX represents matrix multiplication of A and X.

```cpp
template <int M>
struct Matrix {
    int rows, cols;
    vector<vector<int64>> a;

    Matrix() : rows(0), cols(0) {}

    Matrix(int rows, int cols, int64 value = 0)
        : rows(rows), cols(cols), a(rows, vector<int64>(cols, value % M)) {}

    vector<int64>& operator[](int i) {
        return a[i];
    }

    const vector<int64>& operator[](int i) const {
        return a[i];
    }

    static Matrix identity(int n) {
        Matrix I(n, n);

        for (int i = 0; i < n; i++) {
            I[i][i] = 1;
        }

        return I;
    }

    Matrix operator*(const Matrix& other) const {
        assert(cols == other.rows);

        Matrix result(rows, other.cols);

        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < cols; k++) {
                if (a[i][k] == 0) continue;

                for (int j = 0; j < other.cols; j++) {
                    result[i][j] += a[i][k] * other[k][j] % M;

                    if (result[i][j] >= M) {
                        result[i][j] -= M;
                    }
                }
            }
        }

        return result;
    }

    Matrix pow(int64 exponent) const {
        assert(rows == cols);

        Matrix base = *this;
        Matrix result = Matrix::identity(rows);

        while (exponent > 0) {
            if (exponent & 1) {
                result = result * base;
            }

            base = base * base;
            exponent >>= 1;
        }

        return result;
    }
};
```

how to use

```cpp
Matrix<MOD> transition(states, states, 0);
Matrix<MOD> base(states, 1, 0);
base[0][0] = 1;
Matrix<MOD> sol = transition.pow(N) * base;
```

Often times you are solving with a transition matrix, which means it allows you to compute the a_n+1 term from the a_n term.

transition_matrix^power*base_matrix = solution_matrix
