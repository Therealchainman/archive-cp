# Starters 147

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

# Starters 148

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

# Starters 149

## Maximise Sum

### Solution 1: sort, greedy

1. if it is better to invert do that

```cpp

int N;
vector<int> arr;

void solve() {
    cin >> N;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    sort(arr.begin(), arr.end());
    for (int i = 1; i < N; i += 2) {
        if (-arr[i] - arr[i - 1] > arr[i] + arr[i - 1]) {
            arr[i] = -arr[i];
            arr[i - 1] = -arr[i - 1];
        }
    }
    int ans = accumulate(arr.begin(), arr.end(), 0LL);
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

## Chef Loves Beautiful Strings (Easy Version)

### Solution 1:  math, formula

1. Derive the formula to count the number.

```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'

int N;
string S;

int summation(int n) {
    return n * (n + 1) / 2;
}

void solve() {
    cin >> N >> S;
    int x = 0;
    for (int i = 1; i < N; i++) {
        if (S[i] != S[i - 1]) x++;
    }
    int ans = max(0LL, N - x - 1) * x + summation(x - 1);
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

## Kill Monsters (Hard Version)

### Solution 1: math, sort, two pointers

```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'

int N, X, K;
vector<int> arr;

int calc(int cur) {
    int ans = 0;
    for (int x : arr) {
        if (cur > x) {
            ans++;
            cur = x;
        }
    }
    return ans;
}

void solve() {
    cin >> N >> X >> K;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    sort(arr.begin(), arr.end(), greater<int>());
    int base = calc(X);
    int ans = 0;
    int prv = 0;
    vector<int> arr2, health;
    health.push_back(X);
    for (int x : arr) {
        if (X > x) {
            X = x;
            health.push_back(x);
        } else {
            if (x != prv) arr2.push_back(x);
            prv = x;
        }
    }
    int l = 0, r = 0;
    for (int i = 0; i < health.size(); i++) {
        while (r < arr2.size() && arr2[r] >= health[i]) r++;
        while (l < arr2.size() && arr2[l] >= K * health[i]) l++;
        ans = max(ans, base + r - l);
    }
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