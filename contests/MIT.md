# MITIT 

# MITIT 2024 Combined Round

tromino packing, I think it might be inclusion, exclusion principle
But yeah I'm not entirely certain, you'd have to figure out how to appropriately count them though
It is combinatorics related. 


tree coloring I don't know
dp on tree doesn't really work.
I think there may be some greedy idea

connnect buildings
minimum spanning tree
geometry, spatial 
circle
chords on a circle

when you draw a chord on the circle, you prevent anything on one side of the chord from being connected to anything across from the chord. 
so it divides up the remaining problem, cause they are now distinct from each other. 
I think you can divide up N^2 ways though, hmm that is not so great.  There will be N division steps
so N * N^2 is N^3, okay, not great but may pass some subtasks.
Now the question is how do you sum up these answers, take the best answer from each division I suppose. 

At each division you are forming a minimum spanning tree. 
how do you connect two divisions together? 
You need to take one of the two points that create the chord that is the dividing chord.  And for those nodes they need to connect to both a node on both sides and that should work. And just connect it to the node with minimum, okay this is a bit tricky, cause there could be a lot of other chords already formed before you get to this merge step.  And so that would not be easy, you'd have to pick chord so it doesn't intersect with any existing chords.  And still pick the smallest, pick the smallest chord that does not intersect. 

I don't know this is a bit weird, I think I'm off base a bit here.  I don't know if I have enough of an idea to begin implementing.  

In each division connect them back and so on. 

I think that would be N^2 time complexity if you just keep dividing the problem up into smaller and smaller problems. 


## Monotonically Increasing Tardiness Informatics Tournament

### Solution 1:  ceil division

```py
import math
N, M = map(int, input().split())
ans = 0
for _ in range(N):
    a, b = map(int, input().split())
    if a > M: continue
    ans = max(ans, math.ceil((M - a) / b))
print(ans + 1)
```

## Min-Max Game

### Solution 1:  sort, median

```py
N = int(input())
arr = sorted(map(int, input().split()))
print(arr[N // 2])
```

## Tromino Packing

### Solution 1:  dynamic programming

This only solves subproblem

```py
from itertools import product
from collections import Counter
mod = int(1e9) + 7
T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    in_bounds = lambda r, c: 0 <= r < N and 0 <= c < M
    grid = [input() for _ in range(N)]
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    ans = 1
    for r in range(N):
        for c in range(M):
            for dr, dc in product([-1, 1], repeat = 2):
                if not in_bounds(r, c - dc) or grid[r][c - dc] == "#": continue
                if not in_bounds(r + dr, c) or grid[r + dr][c] == "#": continue
                dp[max(r, r + dr) + 1][c + 1] = (dp[max(r, r + dr) + 1][c + 1] + dp[r + 1][min(c, c - dc)]) % mod # add up the ones along the columns
                

    for r, c in product(range(N), range(M)):
        if grid[r][c] == "o":
            cnt = 0
            for dr, dc in product([-1, 1], repeat = 2):
                if not in_bounds(r, c - dc) or grid[r][c - dc] == "#": continue
                if not in_bounds(r + dr, c) or grid[r + dr][c] == "#": continue
                cnt += 1
            ans = (ans * cnt) % mod
    print(ans)
```

## Tree 2-Coloring

### Solution 1:  dp on tree?

```cpp

```

# MITIT 2024 Beginner Round

## A. MITIT

### Solution 1:  string

```py
Q = int(input())
for _ in range(Q):
    s = input()
    ans = False
    for len_ in range(1, len(s)):
        i = len(s) - 2 * len_
        if i <= 0: continue
        B = s[-len_:]
        C = s[i : -len_]
        if B == C: 
            ans = True
            break
    print("YES" if ans else "NO")
```

## B. Taking an Exam

### Solution 1:  math, sort

```py
T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    arr = sorted(map(int, input().split()))
    cnt = 0
    rem = M
    for d in arr:
        if rem - d < 0: break
        rem -= d
        cnt += 1
    print(M + cnt)
```

## C. Delete One Digit

### Solution 1:  divisibility rules, math, conditional logic

```py
def main():
    N = input()
    if N.count("1") == 0: return print(N, 2)
    if N.count("2") == 0:
        if len(N) & 1: N = N.replace("1", "", 1)
        return print(N, 11)
    dsum = sum(map(int, N))
    if dsum % 3 == 1: N = N.replace("1", "", 1)
    if dsum % 3 == 2: N = N.replace("2", "", 1)
    print(N, 3)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Collecting Coins

### Solution 1:  binary search, dijkstra's algorithm

```cpp
const int MAXN = 2e5 + 5;
int N, M;
vector<vector<pair<int, int>>> adj;
int coins[MAXN], rew[MAXN];

bool dijkstra(int cutoff) {
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    vector<int> vis(N, false);
    pq.emplace(0, 0);
    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();
        if (u == N - 1) return true;
        if (vis[u]) continue;
        vis[u] = true;
        for (auto [v, i] : adj[u]) {
            if (cost + coins[i] > cutoff) continue;
            if (rew[i] > coins[i]) return true;
            if (vis[v]) continue;
            pq.emplace(cost + coins[i] - rew[i], v);
        }
    }
    return false;
}

void solve() {
    cin >> N >> M;
    adj.assign(N, {});
    for (int i = 0; i < M; i++) {
        int u, v, c, r;
        cin >> u >> v >> c >> r;
        u--, v--;
        adj[u].emplace_back(v, i);
        adj[v].emplace_back(u, i);
        coins[i] = c;
        rew[i] = r;
    }
    int left = 0, right = 2e14;
    while (left < right) {
        int mid = (left + right) >> 1;
        if (!dijkstra(mid)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    cout << left << endl;
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

# MITIT 2024 Spring Invitational Qualification

## 3-SAT 

### Solution 1: 

only partial results

```py
from collections import Counter
def main():
    N, M = map(int, input().split())
    outdegrees = Counter()
    for i in range(M):
        x, y, z = map(int, input().split())
        x -= 1; y -= 1; z -= 1
        clause = tuple(sorted(set([x, y, z])))
        outdegrees[clause] += 1
    ans = [0] * N
    if M & 1:
        print("YES")
        print(*[1] * N)
    else:
        for c, v in outdegrees.items():
            if v & 1:
                print("YES")
                for i in c:
                    ans[i] = 1
                print(*ans)
                return
        print("NO")
    
if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Busy Marksman

### Solution 1:  greedy, pick from lanes with one target immediately, else pick from rest

```cpp
const int MAXN = 300'005, MAXM = 500'005;
int N, A[MAXN], ans[MAXM];

void solve() {
    cin >> N;
    int sum = 0;
    vector<int> rest, ones;
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        sum += A[i];
        if (A[i] == 1) ones.push_back(i);
        else if (A[i] > 0) rest.push_back(i);
    }
    int i;
    for (i = 1; i <= sum; i++) {
        if (i & 1) {
            if (ones.size()) {
                int v = ones.end()[-1];
                ones.pop_back();
                ans[i - 1] = v + 1;
                A[v]--;
            } else if (rest.size()) {
                int v = rest.end()[-1];
                ans[i - 1] = v + 1;
                A[v]--;
                if (A[v] == 1) {
                    ones.push_back(v);
                    rest.pop_back();
                }
            } else {
                break;
            }
        } else {
            if (rest.size()) {
                int v = rest.end()[-1];
                ans[i - 1] = v + 1;
                A[v]--;
                if (A[v] == 1) {
                    ones.push_back(v);
                    rest.pop_back();
                }
            } else {
                break;
            }
        }
    }

    if (i > sum) {
        cout << "YES" << endl;
        for (int j = 0; j < sum; j++) cout << ans[j] << " ";
        cout << endl;
    } else {
        cout << "NO" << endl;
    }
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

## NM Chars

### Solution 1: 

```py

```

#

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

#

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