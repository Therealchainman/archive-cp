# Additional Problems 2

## Bouncing Balls Steps

### Solution 1:  modular arithmetic, independent variables, lowest common multiple

1. One key is to identify that the position horizontal or vertical are independent of each other. 
1. And the number of corners you hit is based on the lowest common multiple.

```cpp
int64 N, M, K;

int calc(int n) {
    int v = K / n;
    if (v & 1) {
        return n - (K % n);
    }
    return K % n;
}

void solve() {
    cin >> N >> M >> K;
    int64 a = K / (N - 1); // side 
    int64 b = K / (M - 1); // side
    int64 corners = K / lcm<int64>(N - 1, M - 1); // corners
    int64 cnt = a + b - corners;
    int r = calc(N - 1), c = calc(M - 1);
    cout << r + 1 << " " << c + 1 << " " << cnt << endl;
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

## 

### Solution 1: 

```cpp

```