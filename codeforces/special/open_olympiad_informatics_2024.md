# XVIII Open Olympiad in Informatics

## Draw Polygon Lines

### Solution 1:  help me

```py

```

## Evidence Board

### Solution 1:  tree

```cpp

```

## More Gifts

### Solution 1:  binary search, calculating next, coordinate compression

```cpp
#include <bits/stdc++.h>
using namespace std;
 
using i64 = long long;
const int N = 3e5 + 10, B = 60;
 
int n, k, t, a[N], b[N];
 
i64 nxt[B][N];
 
int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
 
    cin >> n >> k >> t;
    for(int i = 0; i < n; i ++)
        cin >> a[i];
 
    {
        vector<int> tmp(n);
        for(int i = 0; i < n; i ++)
            tmp[i] = a[i];
        sort(tmp.begin(), tmp.end());
        auto tend = unique(tmp.begin(), tmp.end());
        if(tend - tmp.begin() <= t) {
            cout << 1 << "\n";
            return 0;
        }
        for(int i = 0; i < n; i ++)
            a[i] = lower_bound(tmp.begin(), tend, a[i]) - tmp.begin();
    }
 
    for(int i = 0; i < n; i ++)
        if(b[a[i]] ++ == 0) t --;
 
    for(int i = n - 1, r = 2 * n - 1; i >= 0; i --) {
        if(b[a[i]] ++ == 0) t --;
        while(t < 0)
            if(-- b[a[(r --) % n]] == 0) t ++;
        nxt[0][i] = r - i + 1;
    }
 
    for(int j = 1; j < B; j ++)
        for(int i = 0; i < n; i ++)
            nxt[j][i] = min(nxt[j - 1][i] + nxt[j - 1][(i + nxt[j - 1][i]) % n], (i64)1e18);
 
    i64 ans = 0, left = (i64)n * k;
    for(int j = B - 1, p = 0; j >= 0; j --)
        if(nxt[j][p] < left) {
            left -= nxt[j][p];
            p = (p + nxt[j][p]) % n;
            ans |= 1ll << j;
        }
    cout << ans + 1;
}
 
```

## 

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

## 

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