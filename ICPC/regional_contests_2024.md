# Regional Contests 2024

## 

### Solution 1: 

```py

```

## Balanced Tree Path

### Solution 1:  dfs, stack, tree

([)]

```cpp
const int INF = 1e9;
int N, ans;
string sym;
const string OPEN = "([{", CLOSE = ")]}";
vector<vector<int>> adj;
stack<int> stk;

void dfs(int u, int p) {
    char br = sym[u];
    char last;
    if (OPEN.find(br) != string::npos) stk.push(br);
    else {
        if (stk.empty()) return;
        last = stk.top();
        if (OPEN.find(last) != CLOSE.find(br)) return;
        stk.pop();
    }
    if (stk.empty()) ans++;
    for (int v : adj[u]) {
        if (v != p) dfs(v, u);
    }
    if (OPEN.find(br) != string::npos) stk.pop();
    else stk.push(last);
}

void solve() {
    cin >> N;
    cin >> sym;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    ans = 0;
    for (int i = 0; i < N; i++) {
        while (!stk.empty()) stk.pop();
        dfs(i, -1);
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    while (T--) {
        solve();
    }
    return 0;
}

```

## ABC String

### Solution 1:  greedy, count

```py
def main():
    s = input()
    CHARS = "ABC"
    counts = [0] * 3
    ans = 0
    for ch in s:
        idx = CHARS.index(ch)
        counts[idx] += 1
        diff = max(counts) - min(counts)
        ans = max(ans, diff)
    print(ans)

if __name__ == '__main__':
    main()
```

## Ordered Problem Set

### Solution 1:  prefix max, brute force

```py
def main():
    n = int(input())
    arr = [int(input()) for _ in range(n)]
    ans = []
    def check():
        pmax = cmax = 0
        sz = n // k
        for i in range(n):
            if i % sz == 0:
                pmax = max(pmax, cmax)
                cmax = 0
            cmax = max(cmax, arr[i])
            if arr[i] < pmax: return False
        return True
    for k in range(2, n + 1):
        if n % k != 0: continue
        if check(): ans.append(k)
    if len(ans) > 0: 
        for num in ans: print(num)
    else: print(-1)

if __name__ == '__main__':
    main()
```

## Item Selection

### Solution 1:  greedy, implementation, math, set, set difference

```py
import math
def main():
    # n items, m per page, page s, p preselected, q to be selected
    n, m, s, p, q = map(int, input().split())
    s -= 1
    minp, maxp = math.inf, -math.inf
    # preselected items
    pre = [0] * p
    # items to be selected
    sel = [0] * q
    for i in range(p):
        x = int(input()) - 1
        pre[i] = x 
    for i in range(q):
        x = int(input()) - 1
        sel[i] = x
    i = j = 0
    while i < p or j < q:
        if i < p and (j == q or pre[i] < sel[j]):
            minp = min(minp, pre[i] // m)
            maxp = max(maxp, pre[i] // m)
            i += 1
        elif j < q and (i == p or sel[j] < pre[i]):
            minp = min(minp, sel[j] // m)
            maxp = max(maxp, sel[j] // m)
            j += 1
        else:
            i += 1
            j += 1
    if minp == math.inf: return print(0)
    ans = i = j = 0
    for page in range((n + m - 1) // m):
        l, r = page * m, min(n, (page + 1) * m)
        pcnt = r - l
        take, rem = set(), set()
        while i < p and pre[i] // m == page: rem.add(pre[i]); i += 1
        while j < q and sel[j] // m == page: take.add(sel[j]); j += 1
        cost = len(take - rem) + len(rem - take)
        if len(take) > 0:
            # deselect all, select or select all, deselect
            cost = min(cost, len(take) + 1, pcnt - len(take) + 1)
        elif len(rem) > 0: # just need to deslect all
            cost = min(cost, 1)
        ans += cost
    ans += maxp - minp
    # traverse the shorter segment a second time
    ans += min(abs(s - minp), abs(s - maxp))
    print(ans)

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py

```

## Streets Behind

### Solution 1:  math

Cannot find the math formula to solve this one. 

```py
import math
def main():
    n, k, a, b = map(int, input().split())
    total = n + k 
    n2 = n
    ans = 0
    g = math.gcd(a, b)
    a //= g
    b //= g
    delta = n * b // a - n 
    rem = (n * b) % a
    if delta <= 0: return print(-1)
    while n < total:
        # 0, 1, 2, 3, 4
        # 0, 2, 4
        # starts at 0, goes till 0 +1
        # starts at 0 goes till 1 +2
        # starts at 1 goes till 2 +3
        # starts at 2 goes till 1 +4
        # starts at 1 goes till 1 +5
        # starts at 1 goes till 2 +6
        # starts at 2 goes till 4 +7
        # starts at 4 goes till 
        # starts at 4, 0, 1, 2, 3, 4, 0 
        # add it to 
        ub = max(1, (total - n) // delta)
        x = min(ub, (a - 1 - rem) // delta + 1)
        n += delta * x
        change = max(1, (rem + delta * x) // a)
        rem = (rem + delta * x) % a
        delta += change
        print("x", x, "n", n, "rem", rem, "delta", delta)
        ans += x
    print(ans)
    ans2 = 0
    while n2 < total:
        floor = n2 * b // a 
        print("delta", floor - n2)
        n2 = floor 
        print("n2", n2)
        ans2 += 1
    print("ans2", ans2)
    assert(ans == ans2)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
"""
1
10 1000000000 13 23
1
10 1000000000 10 12
1
10 1000000000 5 6
"""
```

```py
def main():
    n, k, a, b = map(int, input().split())
    total = n + k 
    n2 = n
    ans = 0
    deltas = []
    while n < total:
        # res.append(n)
        delta_n = n * b // a - n
        if delta_n == 0: return print(-1)
        deltas.append(delta_n)
        left, right = n, total 
        while left < right:
            mid = (left + right) >> 1
            dn = mid * b // a - mid
            if dn <= delta_n:
                left = mid + 1
            else:
                right = mid
        delta = left - n
        cnt = (delta + delta_n - 1) // delta_n
        if cnt == 0: return print(-1)
        n += cnt * delta_n
        ans += cnt
    print(ans)
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

## 

### Solution 1: 

```py

```