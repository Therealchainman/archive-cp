# Atcoder Regular Contest 200-250

# Atcoder Regular Contest 197

## Dot Product

### Solution 1: vector, dot product, geometry, algebra

### Reduce to Two Coordinates

You want to find an ( $X \in \mathbb{R}^n$  ) such that:
$$
A \cdot X > 0 \quad \text{and} \quad B \cdot X < 0.
$$

If all the ratios \( $\frac{A_k}{B_k}$ \) were equal, then \( A \) and \( B \) are collinear, and one half-space would contain the other. So instead, assume that

$$
\frac{A_i}{B_i} > \frac{A_j}{B_j}
$$

for some pair of indices \( i, j \). Equivalently,

$$
A_i B_j > A_j B_i.
$$

Now zero out all other coordinates. Define
$$
X_k = 0 \quad \text{for } k \ne i, j.
$$

Then,

$$
A \cdot X = A_i X_i + A_j X_j, \quad B \cdot X = B_i X_i + B_j X_j.
$$

Choose \( $X_i$ \) and \( $X_j$ \) so that the two dot-products have opposite signs. A neat choice is:

$$
X_i = A_j + B_j, \quad X_j = -(A_i + B_i).
$$

Then,

$$
\begin{aligned}
A \cdot X &= A_i (A_j + B_j) + A_j (-A_i - B_i) \\
          &= A_i A_j + A_i B_j - A_j A_i - A_j B_i \\
          &= A_i B_j - A_j B_i > 0, \\
B \cdot X &= B_i (A_j + B_j) + B_j (-A_i - B_i) \\
          &= B_i A_j + B_i B_j - B_j A_i - B_j B_i \\
          &= B_i A_j - B_j A_i \\
          &= - (A_i B_j - A_j B_i) < 0.
\end{aligned}
$$

Thus, \( $A \cdot X > 0$ \) and \( $B \cdot X < 0$ \) as desired.

By comparing the ratios $A_i/B_i$​ you locate two coordinates where the two linear forms “disagree” in their ordering.

- **Detect non-parallel linear forms.** Checking the ratios $\tfrac{A_k}{B_k}$​​ is a quick test: if they’re not all equal, A and B are not just rescalings of each other.
    
- **Find a separating direction.** Once you know there’s a disagreement, you don’t have to juggle all n coordinates—just those two! You reduce an n-dimensional problem to a little $2\times2$ “trade-off” and get an explicit X that makes one dot-product positive and the other negative.

```cpp
int N;
vector<int64> A, B;

void solve() {
    cin >> N;
    A.assign(N, 0);
    B.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> B[i];
    }
    int idx = -1;
    for (int i = 1; i < N; i++) {
        if (A[i] * B[0] != A[0] * B[i]) {
            idx = i;
            break;
        }
    }
    if (idx == -1) {
        cout << "No" << endl;
        return;
    }
    vector<int64> ans(N, 0);
    ans[0] = A[idx] + B[idx];
    ans[idx] = A[0] + B[0];
    if (A[0] * B[idx] > A[idx] * B[0]) {
        ans[idx] *= -1;
    } else {
        ans[0] *= -1;
    }
    cout << "Yes" << endl;
    for (int x : ans) {
        cout << x << " ";
    }
    cout << endl;
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