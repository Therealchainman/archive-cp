# Convex Hull Trick DP Optimization

Convex hull is an optimization for some dynamic programming problems that satisfy some conditions.  And can take an O(n^2) to O(nlogn) solution often times. 

The simplest form is when the you have a function that is a line, y = mx + b and you are finding the hull of the line.  Then there are variations in techniques when dealing with some conditions.

## Deque optimization

1. The lines can be added in sorted order of slope
2. The query positions are in sorted order

Variation 1:
1. Maximize result, upper convex hull
2. The lines are added in weakly increasing slope
3. The query positions are in weakly increasing order

Example Problems: 
[Commando]<https://dmoj.ca/problem/apio10p1>

These can be solved with a deque to maintain the slopes in sorted order to remove from front and back to maintain only lines that could potentially be optimal solution.

### Commando

```cpp
struct line {
    int m, b;
    int eval(int x) {
        return m * x + b;
    }
};

inline bool overlap(const line &p1, const line &p2, const line &p3) {
  return (p1.m - p2.m) * (p3.b - p1.b) <= (p1.m - p3.m) * (p2.b - p1.b);
}

int n, a, b, c;
deque<line> lines;
vector<int> arr;

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    lines.clear();
    lines.push_back({0, 0});
    cin >> n;
    cin >> a >> b >> c;
    arr.resize(n);
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    // go through queries in sorted order
    // x = psum[j]
    int prev = 0, x = 0;
    for (int i = 0; i < n; i++) {
        x += arr[i];
        while (lines.size() >= 2 && lines[0].eval(x) <= lines[1].eval(x)) {
            lines.pop_front();
        }
        prev = lines[0].eval(x) + a * x * x + b * x + c;
        int m = -2 * a * x;
        int y_intercept = prev + a * x * x - b * x;
        line cur = line{m, y_intercept};
        while (lines.size() >= 2 && overlap(lines.end()[-2], lines.end()[-1], cur)) {
            lines.pop_back();
        }
        lines.push_back(cur);
    }
    cout << prev << endl;
    return 0;
}
```

Example problem 2: [Bear and Bowling 4]<https://codeforces.com/contest/660/problem/F>

this one is non inclusive so make sure you query hull before adding from current position in queries.

```py
from collections import deque

class Line:
    def __init__(self, m, b):
        self.m = m
        self.b = b
    def eval(self, x):
        return self.m * x + self.b
    def __repr__(self):
        return f"slope: {self.m}, y-intercept: {self.b}"

# increasing slopes
def decreasing_monotonic_stack(p1, p2, p3):
    return (p1.m - p2.m) * (p3.b - p1.b) >= (p1.m - p3.m) * (p2.b - p1.b)

def main():
    N = int(input())
    arr = list(map(int, input().split()))
    psum = [0] * (N + 1)
    pval = [0] * (N + 1)
    for i in range(N):
        psum[i + 1] = psum[i] + arr[i]
        pval[i + 1] = pval[i] + (i + 1) * arr[i]
    res = 0
    # increasing slopes
    # dp[i] = max(-i * psum[j] + pval[j]) for j in (i, N] + i * psum[i] - pval[i]
    hull = deque()
    for i in range(N, -1, -1):
        m = -psum[i]
        b = pval[i]
        x = i
        line = Line(m, b)
        # query for position first
        while len(hull) >= 2 and hull[0].eval(x) <= hull[1].eval(x):
            hull.popleft()
        if hull:
            res = max(res, hull[0].eval(x) + i * psum[i] - pval[i])
        # add line into upper convex hull
        while len(hull) >= 2 and decreasing_monotonic_stack(hull[-2], hull[-1], line):
            hull.pop()
        hull.append(line)
    print(res)

if __name__ == '__main__':
    main()
```

Variation 2:
1. Minimize result, lower convex hull
2. The lines are added in weakly decreasing slope
3. The query positions are in weakly increasing order

Solve minimize problem same way as maximize by negating the slope and y-intercept and basically mirroring the graph over the x-axis.

[covered walkway]<https://open.kattis.com/problems/coveredwalkway>

![image](images/dp_optimization_convex_hull/covered_walkway.png)

The one thing that is wrong about the image is that it should be min_j<=i
You can have j == i because you can pick (x - x)^2, it is one of the cases

This is a little weird, because it means you can add the current line into the set to be process at the start and use it.

```cpp
struct line {
    int m, b;
    int eval(int x) {
        return m * x + b;
    }
};

// checks overlap and avoids any doubles
inline bool overlap(const line &p1, const line &p2, const line &p3) {
  return (p1.m - p2.m) * (p3.b - p1.b) <= (p1.m - p3.m) * (p2.b - p1.b);
}

deque<line> lines;
int N, C, x;

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> N >> C;
    int prev = 0;
    for (int i = 0; i < N; i++) {
        cin >> x;
        int m = -2LL * x;
        int b = x * x + prev;
        line cur = {m, b};
        // adding line into set of lines
        while (lines.size() >= 2 && overlap(lines.end()[-2], lines.end()[-1], cur)) {
            lines.pop_back();
        }
        lines.push_back(cur);
        // query the set of lines
        while (lines.size() >= 2 && lines[0].eval(x) >= lines[1].eval(x)) {
            lines.pop_front();
        }
        prev = lines[0].eval(x) + x * x + C;
    }
    cout << prev << endl;
    return 0;
}
```

Variation 3: 


## Another

1. The line are added in sorted order
2. The query positions are in arbitrary order

This requires binary searching to find the line that is optimal for the query position.

Variation 1:
1. maximize, upper convex hull
2. The lines are added in weakly increasing slope
3. The query positions are in arbitrary order

When you have lines [l1, l2] and you are considering to add l3, you need to check that the intersection of l3 with l1 is to the left of intersection of l2 with l1 then you remove l2 from the stack.

Variation 2:
1. maximize, upper convex hull
2. The lines are added in weakly decreasing slope
3. The query positions are in arbitrary order

when you have lines [l1, l2] and you are considering to add l3, you need to check that the intersection of l3 with l1 is to the right of intersection of l2 with l1 then you remove l2 from the stack.

Example problem that contains both of these cases cause it has variation 1 and variation 2 involved is [Product Sum]<https://codeforces.com/contest/631/problem/E>

```py
import math

class Line:
    def __init__(self, m, b):
        self.m = m
        self.b = b
    def eval(self, x):
        return self.m * x + self.b
    def __repr__(self):
        return f"slope: {self.m}, y-intercept: {self.b}"

# increasing slopes
def decreasing_monotonic_stack(p1, p2, p3):
    return (p1.m - p2.m) * (p3.b - p1.b) >= (p1.m - p3.m) * (p2.b - p1.b)

# decreasing slopes
def increasing_monotonic_stack(p1, p2, p3):
    return (p1.m - p2.m) * (p3.b - p1.b) <= (p1.m - p3.m) * (p2.b - p1.b)

def main():
    N = int(input())
    arr = [0] + list(map(int, input().split()))
    psum = [0] * (N + 1)
    total = 0
    for i in range(N):
        psum[i + 1] = psum[i] + arr[i + 1]
        total += (i + 1) * arr[i + 1]
    max_delta = -math.inf
    # increasing slopes
    # dp[i] = max(arr[i] * j - psum[j - 1]) for j in [1, i] + psum[i - 1] - arr[i] * i
    # right cyclic subarray shift
    hull = []
    cmp = lambda i: hull[i].eval(x) < hull[i + 1].eval(x)
    def binary_search(n):
        left, right = 0, n - 1
        while left < right:
            mid = (left + right) >> 1
            if cmp(mid):
                left = mid + 1
            else:
                right = mid
        return left
    # decreasing slopes
    # dp[i] = max(arr[i] * j - psum[j]) for j in [i, N] + psum[i] - arr[i] * i
    # left cyclic subarray shift
    for i in range(1, N + 1):
        m = i
        b = -psum[i - 1]
        x = arr[i]
        line = Line(m, b)
        while len(hull) >= 2 and increasing_monotonic_stack(hull[-2], hull[-1], line):
            hull.pop()
        hull.append(line)
        idx = binary_search(len(hull))
        max_delta = max(max_delta, hull[idx].eval(x) + psum[i - 1] - arr[i] * i)
    hull = []
    for i in range(N, 0, -1):
        m = i
        b = -psum[i]
        x = arr[i]
        line = Line(m, b)
        while len(hull) >= 2 and decreasing_monotonic_stack(hull[-2], hull[-1], line):
            hull.pop()
        hull.append(line)
        idx = binary_search(len(hull))
        max_delta = max(max_delta, hull[idx].eval(x) + psum[i] - arr[i] * i) 
    res = total + max_delta
    print(res)

if __name__ == '__main__':
    main()
```


## Another

1. The lines are added in arbitrary order
2. The query positions are in arbitrary order

I believe this is when you have to use a multiset data structure



Working on this bug for problem product sum

```cpp
/*
first pass
maximum, upper convex hull
The lines are added in weakly increasing slope
maintain a weakly increasing monotonic stack
The query positions are in arbitrary order

second pass
maximum, upper convex hull
the lines are added in weakly decreasing slope
maintain a weakly decreasing monotonic stack
The query positions are in arbitrary order
*/

struct line {
    int m, b;
    int eval(int x) {
        return m * x + b;
    }
};

int N, m, b, x;
vector<int> arr, ints, psum;
deque<line> hull;

inline bool increasing_monotonic_stack(const line &p1, const line &p2, const line &p3) {
  return (p1.m - p2.m) * (p3.b - p1.b) <= (p1.m - p3.m) * (p2.b - p1.b);
}

inline bool decreasing_monotonic_stack(const line &p1, const line &p2, const line &p3) {
    return (p1.m - p2.m) * (p3.b - p1.b) >= (p1.m - p3.m) * (p2.b - p1.b);
}

inline bool intersection_comp(const int i, const int x) {
    return x * (hull[i].m - hull[i + 1].m) >= hull[i + 1].b - hull[i].b;
}

// lower_bound binary search implemented
bool binary_search(int n, int target) {
    int left = 0, right = n - 1;
    while (left < right) {
        int mid = (left + right) >> 1LL;
        if (intersection_comp(mid, target)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> N;
    arr.resize(N + 1);
    psum.assign(N + 1, 0);
    int total = 0;
    for (int i = 0; i < N; i++) {
        cin >> arr[i + 1];
        total += (arr[i + 1] * (i + 1));
        psum[i + 1] = psum[i] + arr[i + 1];
    }
    int max_delta = -1e18;
    hull.clear();
    for (int i = 1; i <= N; i++) {
        m = i;
        b = -psum[i - 1];
        x = arr[i];
        line cur = {m, b};
        while (hull.size() >= 2 && increasing_monotonic_stack(hull.end()[-2], hull.end()[-1], cur)) {
            hull.pop_back();
        }
        hull.push_back(cur);
        int idx = binary_search(hull.size(), x);
        max_delta = max(max_delta, hull[idx].eval(x) + psum[i - 1] - arr[i] * i);
    }
    hull.clear();
    for (int i = N; i > 0; i--) {
        m = i;
        b = -psum[i];
        x = arr[i];
        line cur = {m, b};
        while (hull.size() >= 2 && decreasing_monotonic_stack(hull.end()[-2], hull.end()[-1], cur)) {
            hull.pop_back();
        }
        hull.push_back(cur);
        int idx = binary_search(hull.size(), x);
        max_delta = max(max_delta, hull[idx].eval(x) + psum[i] - arr[i] * i);
    }
    int res = total + max_delta;
    cout << res << endl;
    return 0;
}
```


[Frog 3]<https://atcoder.jp/contests/dp/tasks/dp_z>