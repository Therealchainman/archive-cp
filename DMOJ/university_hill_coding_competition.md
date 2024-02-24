# University Hill Coding Competition

## UHCC1 P2 - String

### Solution 1:  Z algorithm, string matching, deque

```py
def z_algorithm(s: str) -> list[int]:
    n = len(s)
    z = [0]*n
    left = right = 0
    for i in range(1,n):
        # BEYOND CURRENT MATCHED SEGMENT, TRY TO MATCH WITH PREFIX
        if i > right:
            left = right = i
            while right < n and s[right-left] == s[right]:
                right += 1
            z[i] = right - left
            right -= 1
        else:
            k = i - left
            # IF PREVIOUS MATCHED SEGMENT IS NOT TOUCHING BOUNDARIES OF CURRENT MATCHED SEGMENT
            if z[k] < right - i + 1:
                z[i] = z[k]
            # IF PREVIOUS MATCHED SEGMENT TOUCHES OR PASSES THE RIGHT BOUNDARY OF CURRENT MATCHED SEGMENT
            else:
                left = i
                while right < n and s[right-left] == s[right]:
                    right += 1
                z[i] = right - left
                right -= 1
    return z
from collections import deque
def main():
    K = int(input())
    S = input()
    N = int(input())
    counts = [0] * K
    for _ in range(N):
        pat = input()
        m = len(pat)
        z_arr = z_algorithm(pat + "$" + S)
        queue = deque()
        for i in range(K):
            if z_arr[i + m + 1] == m: queue.append(i)
            while len(queue) > 0 and i - queue[0] == m: queue.popleft()
            counts[i] += len(queue)
    ans = sum(1 for x in counts if x > 1)
    print(ans)
if __name__ == '__main__':
    main()
```

```py
def kmp(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]: 
            j = pi[j - 1]
        if s[j] == s[i]: j += 1
        pi[i] = j
    return pi

from collections import deque
def main():
    K = int(input())
    S = input()
    N = int(input())
    counts = [0] * K
    for _ in range(N):
        pat = input()
        m = len(pat)
        encoded = pat + "$" + S
        parr = kmp(encoded)
        queue = deque()
        i = K - 1
        for p in reversed(parr):
            if i < 0: break
            if p == m: queue.append(i)
            while queue and queue[0] - i == m: queue.popleft()
            counts[i] += len(queue)
            i -= 1
    ans = sum(1 for x in counts if x > 1)
    print(ans)
if __name__ == '__main__':
    main()
```

## UHCC1 P3 - Busy Elevator

### Solution 1:  prefix sum and max, suffix min

```py
import math
def main():
    N, L = map(int, input().split())
    arr = list(map(int, input().split()))
    psum = last = pmax = 0
    for i in range(N):
        if psum + arr[i] > L: break
        last = i + 1
        psum += arr[i]
        pmax = max(pmax, arr[i])
    ans = last
    smin = min(arr[last:], default = math.inf)
    if psum + smin <= L: ans += 1
    def pcalc(start, sum_):
        sum_ -= pmax
        res = 0
        for i in range(start, N):
            if sum_ + arr[i] > L: break 
            res = max(res, i)
            sum_ += arr[i]
        return res
    def scalc(start, sum_):
        res = 0
        for i in range(start + 1, N):
            if sum_ + arr[i] > L: break 
            res = max(res, i)
            sum_ += arr[i]
        return res
    ans = max(ans, pcalc(last, psum))
    ans = max(ans, scalc(last, psum))
    print(ans)
if __name__ == '__main__':
    main()
```

## UHCC1 P4 - Manhattan Distance

### Solution 1:  2D, independent 1D, median, sort

```py
def calc(arr, v):
    return sum(abs(arr[i] - v) for i in range(len(arr)))
import math
def main():
    N = int(input())
    xarr, yarr = [0] * N, [0] * N
    for i in range(N):
        x, y = map(int, input().split())
        xarr[i] = x
        yarr[i] = y
    xarr.sort()
    yarr.sort()
    xm, ym = xarr[N // 2], yarr[N // 2]
    ans = math.inf
    med = calc(xarr, xm) + calc(yarr, ym)
    for x, y in [(xm + 1, ym), (xm, ym + 1), (xm - 1, ym), (xm, ym - 1)]:
        ans = min(ans, med + calc(xarr, x) + calc(yarr, y))
    print(ans)

if __name__ == '__main__':
    main()
```

## UHCC1 P5 - Binary Triangles

### Solution 1:  bitset optimization,

given an array of 50,000 elements containing int64, but you just need 0 or 1, you can optimize by using a bitset, so each element is just 1 bit instead of 64  So you can optimize it to 50,000/ 64 = 781 elements.

```cpp
const int MAXN = 405, MAXM = 5e5 + 5;
int N, M;
bitset<MAXM> points[MAXN];
int dist[MAXN][MAXN];

void solve() {
    cin >> N >> M;
    int v;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cin >> v;
            points[i].set(j, v);
        }
    }
    int ans = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            int cnt = (points[i] ^ points[j]).count();
            dist[i][j] = dist[j][i] = cnt;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            for (int k = j + 1; k < N; k++) {
                if (dist[i][j] == dist[i][k] && dist[i][j] == dist[j][k]) {
                    ans++;
                }
            }
        }
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