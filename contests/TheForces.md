# TheForces

# TheForces Round #40 (Maths-Forces)

## Subtractonacci

### Solution 1: math, cancellation, remainder, prefix sum

```cpp
const int64 MOD = 1e9 + 7;
int64 N, A, B;

void solve() {
    cin >> N >> A >> B;
    vector<int64> pref(6);
    pref[0] = 0;
    pref[1] = A;
    pref[2] = B;
    pref[3] = -A + B + MOD;
    pref[4] = -A + MOD;
    pref[5] = -B + MOD;
    for (int i = 1; i < 6; i++) {
        pref[i] = (pref[i] + pref[i - 1]) % MOD;
    }
    int rem = N % 6;
    int64 ans = pref[rem];
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

## Kaosar loves Polynomials

### Solution 1: 

```cpp

```

## Array Forge

### Solution 1: 

```cpp

```

## GCD and LCM in Perfect Sync

### Solution 1: 

```cpp

```

# ???

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```