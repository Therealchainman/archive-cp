# Codeforces Round 923 Div 3

## A. Make it White

### Solution 1:  min and max

```py
def main():
    n = int(input())
    s = input()
    first, last = n, 0
    for i in range(n):
        if s[i] == "B":
            first = min(first, i)
            last = max(last, i)
    print(last - first + 1)
 
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Following the String

### Solution 1:  deque, strings, dictionary of deque

```py
from collections import defaultdict
import string
 
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    ans = [None] * n
    letters = defaultdict(list)
    letters[0] = list(string.ascii_lowercase)
    for i in range(n):
        ch = letters[arr[i]].pop()
        ans[i] = ch
        letters[arr[i] + 1].append(ch)
    print("".join(ans))
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Choose the Different Ones!

### Solution 1:  set, set intersection, set difference

```py
def main():
    n, m, k = map(int, input().split())
    A = set(filter(lambda x: 1 <= x <= k, map(int, input().split())))
    B = set(filter(lambda x: 1 <= x <= k, map(int, input().split())))
    shared = A & B
    arr1, arr2 = list(A - shared), list(B - shared)
    for x in shared:
        if len(arr1) < k // 2: arr1.append(x)
        else: arr2.append(x)
    ans = "YES" if len(arr1) == len(arr2) == k // 2 else "NO"
    print(ans)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Find the Different Ones!

### Solution 1:  sort, line sweep, set, distinct adjacent always

probably could binary search as well

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    q = int(input())
    starts, ends = [None] * q, [None] * q
    ans = [[-1] * 2 for _ in range(q)]
    for i in range(q):
        l, r = map(int, input().split())
        l -= 1; r -= 1
        starts[i] = (l, i)
        ends[i] = (r, i)
    starts.sort(); ends.sort()
    cur = set()
    s = e = 0
    for i in range(n):
        if i > 0 and arr[i] != arr[i - 1]:
            for j in cur:
                ans[j][0] = i
                ans[j][1] = i + 1
            cur.clear()
        while s < len(starts) and starts[s][0] == i:
            cur.add(starts[s][1])
            s += 1
        while e < len(ends) and ends[e][0] == i:
            cur.discard(ends[e][1])
            e += 1
    for i in range(q):
        print(*ans[i])
    print()

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Clever Permutation

### Solution 1:  greedy

```py
def main():
    n, k = map(int, input().split())
    ans = [0] * (n + 1)
    start, end = 1, n
    for i in range(k // 2):
        for j in range(2 * i + 1, n + 1, k):
            ans[j] = start
            start += 1
        for j in range(2 * i + 2, n + 1, k):
            ans[j] = end
            end -= 1
    print(*ans[1:])
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Microcycle

### Solution 1:  maximum spanning tree, union find, dfs, tree

```cpp
int N, M, src, dst, minw;
vector<int> path;
vector<vector<int>> adj;

struct Edge {
    int u, v, w;
};

struct UnionFind {
    vector<int> parents, size;
    void init(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=find(parents[i]);
    }

    bool union_(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
            return true;
        }
        return false;
    }
};

bool dfs(int u, int p) {
    path.push_back(u + 1);
    if (u == dst) return true;
    for (int v : adj[u]) {
        if (v == p) continue;
        if (dfs(v, u)) return true;
    }
    path.pop_back();
    return false;
}

void solve() {
    cin >> N >> M;
    vector<Edge> edges(M);
    adj.assign(N, vector<int>());
    int u, v, w;
    for (int i = 0; i < M; i++) {
        cin >> u >> v >> w;
        u--; v--;
        edges[i] = {u, v, w};
    }
    sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.w > b.w; // Descending order
    });
    UnionFind dsu;
    dsu.init(N);
    minw = INT_MAX;
    for (auto &[u, v, w] : edges) {
        if (!dsu.union_(u, v)) {
            minw = w; src = u; dst = v;
        } else {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }
    path.clear();
    dfs(src, -1);
    cout << minw << " " << path.size() << endl;
    for (int u : path) {
        cout << u << " ";
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

## G. Paint Charges

### Solution 1: 

```py

```

