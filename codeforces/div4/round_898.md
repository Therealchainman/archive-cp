# Codeforces Round 898 Div 4

## A. Short Sort

### Solution 1:  loop

```py
from itertools import product

def main():
    s = list(input())
    n = len(s)
    target = ["a", "b", "c"]
    for i, j in product(range(n), repeat = 2):
        s[i], s[j] = s[j], s[i]
        if s == target: return print("YES")
        s[i], s[j] = s[j], s[i]
    print("NO")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Good Kid

### Solution 1:  math + greedy

```py
import math

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    arr.sort()
    arr[0] += 1
    print(math.prod(arr))

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Target Practice

### Solution 1:  math + grid

```py
from itertools import product

def main():
    n = 10
    grid = [list(input()) for _ in range(n)]
    res = 0
    for r, c in product(range(n), repeat = 2):
        if grid[r][c] != "X": continue
        res += min(r, c, n - r - 1, n - c - 1) + 1
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. 1D Eraser

### Solution 1:  fixed sliding window

```py
def main():
    n, k = map(int, input().split())
    cells = input()
    left = res = 0
    while left < n:
        if cells[left] == "W": 
            left += 1
            continue
        res += 1
        right = min(n, left + k)
        left = right
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Building an Aquarium

### Solution 1:  binary search

```py
def main():
    n, x = map(int, input().split())
    arr = list(map(int, input().split()))
    left, right = 0, 10**11
    while left < right:
        mid = (left + right + 1) >> 1
        if sum(max(0, mid - a) for a in arr) <= x:
            left = mid
        else:
            right = mid - 1
    print(left)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Money Trees

### Solution 1:  two sliding windows 

sliding window for the subarray satisfying the height constraint
sliding window for the longest subarray within that window. 

```py
def main():
    n, k = map(int, input().split())
    values = list(map(int, input().split()))
    height = list(map(int, input().split()))
    res = left = 0
    def window(left, right):
        len_ = 0
        s = 0
        i = left
        while i < right:
            s += values[i]
            while left <= i and s > k:
                s -= values[left]
                left += 1
            len_ = max(len_, i - left + 1) 
            i += 1
        return len_
    while left < n:
        right = left + 1
        while right < n and height[right - 1] % height[right] == 0:
            right += 1
        res = max(res, window(left, right))
        left = right
    print(res)


if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G. ABBC or BACB

### Solution 1:  greedy + ad hoc

the count of B can at worse be one less than number of A segments.  A segment is just a continuous segment with A, such as AAA
if B is one less than A segments then that means you cannot cover all the A segment, and exactly one A segment will not be covered, so just subtract the smallest A segment from the total count A elements. 

if count of B is greater than or equal to number of A segments, you can cover all A segments.

```py
from itertools import groupby
import math

def main():
    s = input()
    AG, BG = [], []
    acount = bcount = 0
    min_a = math.inf
    for key, grp in groupby(s):
        len_ = len(list(grp))
        if key == "A":
            AG.append(len_)
            acount += len_
            min_a = min(min_a, len_)
        else:
            BG.append(len_)
            bcount += len_
    res = acount - (min_a if bcount < len(AG) else 0)
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## H. Mad City

### Solution 1:  bfs + topological sort + cycle + undirected graph

undirected graph with exactly one cycle, basically need to find if b can make it to cycle before a can. 

```py
from collections import deque

def main():
    n, a, b = map(int, input().split())
    a -= 1
    b -= 1
    adj_list = [[] for _ in range(n)]
    degrees = [0] * n
    for _ in range(n):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        degrees[u] += 1
        degrees[v] += 1
        adj_list[u].append(v)
        adj_list[v].append(u)
    dq = deque()
    for i in range(n):
        if degrees[i] == 1:
            dq.append(i)
            degrees[i] = 0
    while dq:
        u = dq.popleft()
        for v in adj_list[u]:
            if degrees[v] == 0: continue
            degrees[v] -= 1
            if degrees[v] == 1:
                dq.append(v)
                degrees[v] = 0
    dq = deque([(b, 0)])
    vis = [0] * n
    vis[b] = 1
    while dq:
        u, dist = dq.popleft()
        if degrees[u] > 0:
            dist_b = dist
            b = u
            break
        for v in adj_list[u]:
            if vis[v]: continue
            vis[v] = 1
            dq.append((v, dist + 1))
    dq = deque([(a, 0)])
    vis = [0] * n
    vis[a] = 1
    while dq:
        u, dist = dq.popleft()
        if u == b:
            dist_a = dist
            break
        for v in adj_list[u]:
            if vis[v]: continue
            vis[v] = 1
            dq.append((v, dist + 1))
    res = "YES" if dist_b < dist_a else "NO"
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

```cpp
int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    int t = read();
    while (t--) {
        int n = read(), a = read(), b = read();
        a--;
        b--;
        vector<vector<int>> adj_list(n);
        for (int i = 0; i < n; i++) {
            int u = read(), v = read();
            u--;
            v--;
            adj_list[u].push_back(v);
            adj_list[v].push_back(u);
        }
        vector<int> vis(n, 0);
        vector<int> parents(n, -1);
        vector<bool> cycle(n, false);
        function<void(int, int)> dfs = [&](int u, int parent) {
            vis[u] = 1;
            for (int v : adj_list[u]) {
                if (v == parent) continue;
                if (vis[v] == 1) {
                    cycle[v] = true;
                    int node = u;
                    while (node != v) {
                        cycle[node] = true;
                        node = parents[node];
                    }
                }
                if (vis[v]) continue;
                parents[v] = u;
                dfs(v, u);
            }
            vis[u] = 2;
        };
        dfs(0, -1);
        function<pair<int, int>(int)> sdist = [&](int start) {
            deque<int> dq;
            dq.push_back(start);
            vector<bool> vis(n, false);
            int depth = 0;
            while (!dq.empty()) {
                int sz = dq.size();
                while (sz--) {
                    int u = dq.front();
                    dq.pop_front();
                    if (cycle[u]) return make_pair(u, depth);
                    for (int v : adj_list[u]) {
                        if (vis[v]) continue;
                        vis[v] = true;
                        dq.push_back(v);
                    }
                }
                depth += 1;
            }
            return make_pair(n, 0LL);
        };

        function<int(int, int)> dist2 = [&](int start, int end) {
            deque<int> dq;
            dq.push_back(start);
            vector<bool> vis(n, false);
            vis[start] = true;
            int depth = 0;
            while (!dq.empty()) {
                int sz = dq.size();
                while (sz--) {
                    int u = dq.front();
                    dq.pop_front();
                    if (u == end) return depth;
                    for (int v : adj_list[u]) {
                        if (vis[v]) continue;
                        vis[v] = true;
                        dq.push_back(v);
                    }
                }
                depth += 1;
            }
            return n;
        };
        pair<int, int> result = sdist(b);
        int node_b = result.first;
        int dist_b = result.second;
        int dist_a = dist2(a, node_b);
        if (a != b && (dist_b == 0 || dist_b < dist_a)) {
            cout << "YES" << endl;
        } else {
            cout << "NO" << endl;
        }                                                           
    }
    return 0;
}
```

