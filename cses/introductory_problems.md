# Introductory Problems

## Raab Game I

### Solution 1: constructive algorithm, greedy

```cpp
int N, A, B;

void output(const vector<int> &arr) {
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;
}

void solve() {
    cin >> N >> A >> B;
    vector<int> P1(N), P2(N);
    iota(P1.begin(), P1.end(), 1);
    for (int i = 0, j = 0; i < N; i++) {
        if (i < B) {
            P2[i] = P1[i] + A;
        } else if (j < A) {
            P2[i] = ++j;
        } else {
            P2[i] = P1[i];
        }
    }
    for (int i = 0; i < N; i++) {
        if (P1[i] > P2[i]) A--;
        else if (P1[i] < P2[i]) B--;
        if (P2[i] > N) {
            cout << "NO" << endl;
            return;
        }
    }
    if (A != 0 || B != 0) {
        cout << "NO" << endl;
        return;
    }
    cout << "YES" << endl;
    output(P1);
    output(P2);
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