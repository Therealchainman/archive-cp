# Polya Enumeration Theorem

## Example 1: Counting Necklaces

Read understanding cycles in rotations.  

And solve this with the formula: 

C(πᵢ) = gcd(i, n)

Where πᵢ is the rotation by i positions, and n is the length of the necklace. 

Then the number of distinct necklaces is given by: 

(1/n) * Σ (k=1 to n) m^(gcd(k, n))

Where m is the number of colors.

```cpp
const int MOD = 1e9 + 7;
int N, M;
vector<int64> powM;

int64 inv(int i, int64 m) {
  return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

void solve() {
    cin >> N >> M;
    powM.assign(N + 1, 1);
    for (int i = 1; i <= N; ++i) {
        powM[i] = powM[i - 1] * M % MOD;
    }
    int64 ans = 0;
    for (int i = 1; i <= N; ++i) {
        int g = gcd(N, i);
        ans = (ans + powM[g]) % MOD;
    }
    ans = ans * inv(N, MOD) % MOD;
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

