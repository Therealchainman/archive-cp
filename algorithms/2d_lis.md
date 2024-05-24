# 2D LIS

LIS = Longest Increasing Subsequence, it is not described here, read more online to understand what that means.  

For the 2D problem for LIS or LIS for pairs, you can order one pair after another if (x2, y2) comes after (x1, y1) in the array and x2 > x1 and y2 > y1.  

## Disjoint Staircase Algorithm in CPP

This is the array of disjoint staircase algorithm that solves the problem in O(N(log(N))^2)

It is a little bit similar to the patience algorithm that solves the 1D LIS problem in O(NlogN).  The difference is it had to be extended to 2 dimensional space.

```cpp
int N;
vector<pair<int, int>> points;

struct StairCase {
    map<int, int> M;
    bool find(int x, int y) {
        auto it = M.lower_bound(x);
        if (it != M.begin() && y > prev(it) -> second) return true;
        return false;
    }
    void add(int x, int y) {
        auto start_it = M.lower_bound(x);
        auto end_it = start_it;
        while (end_it != M.end() && y <= end_it -> second) end_it++;
        M.erase(start_it, end_it);
        M.insert({x, y});
    }
};

vector<StairCase> stairs;

void solve() {
    cin >> N;
    points.resize(N);
    for (int i = 0; i < N; i++) {
        int x, y;
        cin >> x >> y;
        points[i] = {x, y};
    }
    int L = 1;
    stairs.assign(N + 1, StairCase());
    stairs[1].add(points[0].first, points[0].second);
    for (int i = 1; i < N; i++) {
        auto [x, y] = points[i];
        int lo = 1, hi = L + 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (stairs[mid].find(x, y)) lo = mid + 1;
            else hi = mid;
        }
        stairs[lo].add(x, y);
        L = max(L, lo);
    }
    cout << L << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Visualization of algorithm

Run this in a jupyter notebook, need to split these into separate cells. 

Fix any missing imports

```py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from sortedcontainers import SortedDict
```

```py
%matplotlib notebook
```

```py
n = 200
num = 30
arr = [(random.randint(0, num), random.randint(0, num)) for _ in range(n)]
print(arr)
class Frame:
    def __init__(self, lis_group, x, y, is_add):
        self.lis_group = lis_group
        self.x = x
        self.y = y
        self.is_add = is_add
frames = []
# O(n^2) slow approach to confirm faster algorithm works. 
def longest_increasing_subsequence(pairs):
    n = len(pairs)
    dp = [1] * n
    max_length = 1
    for i in range(1, n):
        for j in range(i):
            if pairs[i][0] > pairs[j][0] and pairs[i][1] > pairs[j][1]:
                dp[i] = max(dp[i], dp[j] + 1)
        max_length = max(max_length, dp[i])
    return max_length
class StairCase:
    def __init__(self, length):
        self.map = SortedDict()
        self.length = length
    def find(self, x, y):
        idx = self.map.bisect_left(x)
        if idx == 0: return False
        idx -= 1
        if self.map.peekitem(idx)[1] < y: return True
        return False
    def add(self, x, y):
        idx = self.map.bisect_left(x)
        fr = []
        while idx < len(self.map) and y <= self.map.peekitem(idx)[1]:
            pair = self.map.popitem(idx)
            fr.append(Frame(self.length, *pair, False))
        cur = self.map.get(x, math.inf)
        if y < cur:
            fr.append(Frame(self.length, x, y, True))
        frames.append(fr)
        self.map[x] = min(self.map.get(x, math.inf), y)
def lis_2d(pairs):
    stairs = [StairCase(i) for i in range(len(pairs))]
    stairs[1].add(*pairs[0])
    L = 1
    for x, y in pairs[1:]:
        l, r = 0, L
        while l < r:
            m = (l + r + 1) >> 1
            if stairs[m].find(x, y):
                l = m
            else:
                r = m - 1
        if l >= L:
            L += 1
        stairs[l + 1].add(x, y)
    return L
x = longest_increasing_subsequence(arr)
lis = lis_2d(arr)
print(x, lis)
assert(x == lis)
fig, ax = plt.subplots()
data = [[(0, num), (num, num)] for _ in range(lis + 1)]
colors = ["w", "r", "g", "b", "y", "m", "c", "k", "k", "k", "k"]
ln = [plt.plot([], [], colors[i], drawstyle = "steps-post", label = str(i))[0]  for i in range(lis + 1)]
ax.set_xlim(0, num + 10)
ax.set_ylim(0, num + 10)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("2D LIS Multiverse Staircase Algorithm Visualization")
ax.legend(loc = "upper right")
def init(): # why is this necessary?
    for i in range(1, lis + 1):
        ln[i].set_data([], [])
    return ln
def update(frames):
    for frame in frames:
        if frame.is_add:
            data[frame.lis_group].append((frame.x, frame.y))
            data[frame.lis_group].sort()
        else:
            data[frame.lis_group].remove((frame.x, frame.y))
        data[frame.lis_group][-1] = (num, data[frame.lis_group][-2][1])
    for i in range(1, lis + 1):
        ln[i].set_data([x for x, _ in data[i]], [y for _, y in data[i]])
    return ln[1:]

ani = animation.FuncAnimation(fig, update, frames = frames, init_func = init, blit = True, repeat = False)
HTML(ani.to_jshtml())
```