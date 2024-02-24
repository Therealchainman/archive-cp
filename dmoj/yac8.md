# Yet Another Contest 8

## Permutation Sorting

### Solution 1:  prefix max, greedy

minimize range of sorting, the only elements that don't need to be sorted are those at their corresponding index, and no element prior belongs after it.

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    pmax = cnt = 0
    for i, num in enumerate(arr, start = 1):
        if num == i and pmax < num: cnt += 1
        pmax = max(pmax, num)
    ans = n - cnt
    print(ans)
if __name__ == '__main__':
    main()
```

## No More Modern Art

### Solution 1:  counter, xor

If you do the math the xors cancel and you are really just trying to find if num ^ x exists in the array.  If it does then you can xor num with that element to get x. 

recall a ^ b = x => a ^ x = b, so we are just finding if ths b exists in the array.

```py
from collections import Counter
def main():
    n, x = map(int, input().split())
    arr = list(map(int, input().split()))
    counts = Counter(arr)
    for num in arr:
        counts[num] -= 1
        target = num ^ x
        if counts[target] > 0: return print("YES")
        counts[num] += 1
    print("NO")
if __name__ == '__main__':
    main()
```

## Herobrine

### Solution 1:  small-to-large merging technique, dfs, tree

```cpp
vector<vector<int>> adj;
vector<unordered_map<int, int>> freq;
vector<unordered_map<int, int>> greater_freq;
vector<int> ans, sz;
vector<vector<int>> ores;

void dfs(int u, int p) {
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
        if (sz[v] > sz[u]) {
            swap(sz[u], sz[v]);
            swap(freq[u], freq[v]);
            swap(greater_freq[u], greater_freq[v]);
            ans[u] = max(ans[u], ans[v]);
        }
        sz[u] += sz[v];
        for (auto [o, f] : freq[v]) {
            for (int i = 0; i < f; ++i) {
                freq[u][o]++;
                greater_freq[u][freq[u][o]]++;
                ans[u] = max(ans[u], greater_freq[u][freq[u][o]] * freq[u][o]);
            }
        }
    }
    for (int o : ores[u]) {
        sz[u]++;
        freq[u][o]++;
        greater_freq[u][freq[u][o]]++;
        ans[u] = max(ans[u], greater_freq[u][freq[u][o]] * freq[u][o]);
    }
}

signed main() {
    int n;
    cin >> n;
    vector<int> parent(n);
    for (int& p : parent) {
        cin >> p;
    }
    adj.assign(n + 1, vector<int>());
    for (int i = 0; i < n; ++i) {
        adj[parent[i]].push_back(i + 1);
        adj[i + 1].push_back(parent[i]);
    }
    ores.assign(n + 1, vector<int>());
    for (int i = 1; i <= n; ++i) {
        int m;
        cin >> m;
        ores[i].resize(m);
        for (int j = 0; j < m; ++j) {
            cin >> ores[i][j];
        }
    }
    sz.assign(n + 1, 0);
    freq.assign(n + 1, unordered_map<int, int>());
    greater_freq.assign(n + 1, unordered_map<int, int>());
    ans.resize(n + 1);
    dfs(0, -1);
    for (int i = 1; i <= n; ++i) {
        cout << ans[i] << endl;
    }
    return 0;
}
```

## Fluke 2

### Solution 1: 

I think I'm close to solution that get's lot of credit meh. 

```py
import sys
from itertools import product
def play1(N, M):
    print(1, flush = True) # player 1
    grid = [[0] * (M + 1) for _ in range(N + 1)]
    print(1, 1, flush = True)
    grid[1][1] ^= 1
    progress = input()
    if progress != "C": return progress
    r, c = map(int, input().split())
    grid[r][c] ^= 1
    progress = input()
    sys.stdout.flush()
    if progress != "C": return progress
    while True:
        if r > 1 and r < N and grid[r - 1][c]:
            r += 1
        elif c > 1 and c < M and grid[r][c - 1]:
            c += 1
        elif r < N and r > 1 and grid[r + 1][c]:
            r -= 1
        elif c < M and c > 1 and grid[r][c + 1]:
            c -= 1
        elif r > 1:
            r -= 1
        else: c -= 1
        print(r, c, flush = True)
        grid[r][c] ^= 1
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
        r1, c1 = map(int, input().split())
        grid[r1][c1] ^= 1
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
def play2(N, M):
    print(2, flush = True) # player 2
    grid = [[0] * (M + 1) for _ in range(N + 1)]
    r, c = map(int, input().split())
    grid[r][c] ^= 1
    progress = input()
    sys.stdout.flush()
    if progress != "C": return progress
    for r1, c1 in product(range(1, N + 1), range(1, M + 1)):
        if (r1, c1) == (r, c): continue
        print(r1, c1, flush = True)
        grid[r1][c1] ^= 1
        break
    pr, pc = r, c
    progress = input()
    sys.stdout.flush()
    if progress != "C": return progress
    while True:
        r, c = map(int, input().split())
        grid[r][c] ^= 1
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
        if r > 1 and r < N and grid[r - 1][c]:
            print(r - 1, c, flush = True)
        elif c > 1 and c < M and grid[r][c - 1]:
            print(r, c - 1, flush = True)
        elif r < N and r > 1 and grid[r + 1][c]:
            print(r + 1, c, flush = True)
        elif c < M and c > 1 and grid[r][c + 1]:
            print(r, c + 1, flush = True)
        else:
            print(pr, pc, flush = True)
            progress = input()
            if progress != "C": return progress
            pr, pc = r, c
def main():
    N, M, T = map(int, input().split())
    for _ in range(T):
        if N <= 2 and M == 2: 
            if play2(N, M) != "W": break
        elif N <= 2:
            if play3(N, M) != "W": break
        else: 
            if play1(N, M) != "W": break
    
if __name__ == '__main__':
    main()
```

```py
import sys
from itertools import product
import math
def play1(N, M):
    print(1, flush = True) # player 1
    blocks = set()
    in_bounds = lambda r, c: 1 <= r <= N and 1 <= c <= M
    neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    manhattan_distance = lambda r1, c1, r2, c2: abs(r1 - r2) + abs(c1 - c2)
    # determine two blocks are adjacent
    def winning_move():
        for r, c in blocks:
            # check vertical
            adj = False
            pair = None
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or nc != c: continue
                if (nr, nc) in blocks: adj = True
                elif (nr, nc) != prev: pair = (nr, nc)
            if adj and pair is not None: return pair
            # check horizontal
            adj = False
            pair = None
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or nr != r: continue
                if (nr, nc) in blocks: adj = True
                elif (nr, nc) != prev: pair = (nr, nc)
            if adj and pair is not None: return pair
        return None, None
    def move():
        r1, c1 = blocks.pop()
        r2, c2 = blocks.pop()
        nearest = math.inf
        best_pair = None
        for nr, nc in neighborhood(r1, c1):
            if not in_bounds(nr, nc) or (nr, nc) == (r2, c2) or (nr, nc) == prev: continue
            d = manhattan_distance(nr, nc, r2, c2)
            if d < nearest:
                nearest = d
                best_pair = (nr, nc)
        blocks.update([(r1, c1), (r2, c2)])
        return best_pair
    prev = None
    while True:
        r, c = winning_move()
        if r is None:
            if not blocks: # 0 blocks
                r, c = 1, 1
            else: # 2 blocks
                r, c = move()
        print(r, c, flush = True)
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
        blocks.add((r, c))
        r, c = map(int, input().split())
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
        if (r, c) in blocks:
            blocks.remove((r, c))
        else:
            blocks.add((r, c))
        prev = (r, c)
def play2(N, M):
    print(2, flush = True) # player 2
    while True:
        r, c = map(int, input().split())
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
        for nr, nc in product(range(1, N + 1), range(1, M + 1)):
            if (nr, nc) == (r, c): continue
            print(nr, nc, flush = True)
            break
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
def main():
    N, M, T = map(int, input().split())
    for _ in range(T):
        if M == 2: 
            if play2(N, M) != "W": break
        else: 
            if play1(N, M) != "W": break

if __name__ == '__main__':
    main()
```

```py
import sys
from itertools import product
import math
def play1(N, M):
    print(1, flush = True) # player 1
    blocks = set()
    in_bounds = lambda r, c: 1 <= r <= N and 1 <= c <= M
    neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    manhattan_distance = lambda r1, c1, r2, c2: abs(r1 - r2) + abs(c1 - c2)
    # determine two blocks are adjacent
    def winning_move():
        for r, c in blocks:
            # check vertical
            adj = False
            pair = None
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or nc != c: continue
                if (nr, nc) in blocks: adj = True
                elif (nr, nc) != prev: pair = (nr, nc)
            if adj and pair is not None: return pair
            # check horizontal
            adj = False
            pair = None
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or nr != r: continue
                if (nr, nc) in blocks: adj = True
                elif (nr, nc) != prev: pair = (nr, nc)
            if adj and pair is not None: return pair
        return None, None
    def get(r1, c1, r2, c2):
        nearest = math.inf
        best_pair = None
        for nr, nc in neighborhood(r1, c1):
            if not in_bounds(nr, nc) or (nr, nc) == (r2, c2) or (nr, nc) == prev: continue
            d = manhattan_distance(nr, nc, r2, c2)
            if d < nearest:
                nearest = d
                best_pair = (nr, nc)
        return best_pair
    def move():
        r1, c1 = blocks.pop()
        r2, c2 = blocks.pop()
        best_pair = get(r1, c1, r2, c2)
        if best_pair is None: best_pair = get(r2, c2, r1, c1)
        blocks.update([(r1, c1), (r2, c2)])
        return best_pair
    prev = None
    while True:
        r, c = winning_move()
        if r is None:
            if not blocks: # 0 blocks
                r, c = (N + 1) // 2, (M + 1) // 2
            else: # 2 blocks
                r, c = move()
        print(r, c, flush = True)
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
        blocks.add((r, c))
        r, c = map(int, input().split())
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
        if (r, c) in blocks:
            blocks.remove((r, c))
        else:
            blocks.add((r, c))
        prev = (r, c)
def play2(N, M):
    print(2, flush = True) # player 2
    while True:
        r, c = map(int, input().split())
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
        for nr, nc in product(range(1, N + 1), range(1, M + 1)):
            if (nr, nc) == (r, c): continue
            print(nr, nc, flush = True)
            break
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
def play3(N, M):
    print(2, flush = True) # player 2
    blocks = set()
    def find():
        row_counts = [0] * (N + 1)
        for r, _ in blocks:
            row_counts[r] += 1
        for r in range(1, N + 1):
            if row_counts[r] == 0:
                for c in range(1, M + 1):
                    if (r, c) != prev: return (r, c)
            elif row_counts[r] == 2:
                for c in range(1, M + 1):
                    if (r, c) in blocks and (r, c) != prev: return (r, c)
    prev = None
    while True:
        r, c = map(int, input().split())
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
        if (r, c) in blocks:
            blocks.remove((r, c))
        else:
            blocks.add((r, c))
        prev = (r, c)
        # print("blocks", blocks, prev, flush = True)
        r, c = find()
        print(r, c, flush = True)
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
        if (r, c) in blocks:
            blocks.remove((r, c))
        else:
            blocks.add((r, c))
        # print("blocks", blocks, flush = True)

def main():
    N, M, T = map(int, input().split())
    for _ in range(T):
        if M == 2:
            if play2(N, M) != "W": break
        elif N == 2:
            if play3(N, M) != "W": break
        else: 
            if play1(N, M) != "W": break

if __name__ == '__main__':
    main()
```

## Hidden Tree

### Solution 1:

try this

```py

```