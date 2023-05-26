
```cpp
#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}
```

## B. Introduction: Convex Hull Trick Optimization

### Solution 1:  optimize dp with convex hull trick

```cpp
const long long is_query = -(1LL<<62);
struct line {
    long long m, b;
    mutable function<const line*()> succ;
    bool operator<(const line& rhs) const {
        if (rhs.b != is_query) return m < rhs.m;
        const line* s = succ();
        if (!s) return 0;
        long long x = rhs.m;
        return b - s->b < (s->m - m) * x;
    }
};

struct dynamic_hull : public multiset<line> { // will maintain upper hull for maximum
    const long long inf = LLONG_MAX;
    bool bad(iterator y) {
        auto z = next(y);
        if (y == begin()) {
            if (z == end()) return 0;
            return y->m == z->m && y->b <= z->b;
        }
        auto x = prev(y);
        if (z == end()) return y->m == x->m && y->b <= x->b;

		/* compare two lines by slope, make sure denominator is not 0 */
        long long v1 = (x->b - y->b);
        if (y->m == x->m) v1 = x->b > y->b ? inf : -inf;
        else v1 /= (y->m - x->m);
        long long v2 = (y->b - z->b);
        if (z->m == y->m) v2 = y->b > z->b ? inf : -inf;
        else v2 /= (z->m - y->m);
        return v1 >= v2;
    }
    void insert_line(long long m, long long b) {
        auto y = insert({ m, b });
        y->succ = [=] { return next(y) == end() ? 0 : &*next(y); };
        if (bad(y)) { erase(y); return; }
        while (next(y) != end() && bad(next(y))) erase(next(y));
        while (y != begin() && bad(prev(y))) erase(prev(y));
    }
    long long eval(long long x) {
        auto l = *lower_bound((line) { x, is_query });
        return l.m * x + l.b;
    }
};

int main(){
	int n = read();
	vector<long long> a(n), b(n), dp(n), prefix_sum(n);
	long long pref = 0;
	for (int i = 0; i < n; ++i) {
		a[i] = read();
		pref += a[i];
		prefix_sum[i] = pref;
	}
	for (int i = 0; i < n; ++i) b[i] = read();
	dynamic_hull cht;
	cht.insert_line(0, 0);
	for (int i = 1; i < n; i++) {
		dp[i] = prefix_sum[i] * b[i] - cht.eval(b[i]);
		cht.insert_line(prefix_sum[i - 1], -dp[i]);
	}
	for (int i = 0; i < n; ++i) cout << dp[i] << " ";
}
```

## A. Introduction: Segment Tree Optimization

### Solution 1:  segment tree optimization for dp + min segment tree

```py
import math

class SegmentTree:
    def __init__(self, n: int, neutral: int, func, initial_arr):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)] 
        self.build(initial_arr)

    def build(self, initial_arr: List[int]) -> None:
        for i, segment_idx in enumerate(range(self.n)):
            segment_idx += self.size - 1
            val = initial_arr[i]
            self.nodes[segment_idx] = val
            self.ascend(segment_idx)

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx])
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.nodes[segment_idx] = val
        self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.func(result, self.nodes[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"nodes array: {self.nodes}, next array: {self.nodes}"
    
from itertools import accumulate

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    prefix_sum = list(accumulate(arr, initial = 0))
    ranges = [0]*n
    for i in range(1, n):
        x, y = map(int, input().split())
        x -= 1
        y -= 1
        ranges[i] = (x, y)
    st = SegmentTree(n + 1, math.inf, min, [-x for x in prefix_sum])
    dp = [0]*n
    for i in range(1, n):
        left, right = ranges[i]
        dp[i] = st.query(left, right + 1) + prefix_sum[i + 1]
        st.update(i, dp[i] - prefix_sum[i])
    return ' '.join(map(str, dp))

if __name__ == '__main__':
    print(main())
```

##

### Solution 1: 

```cpp

```

## E. Kalila and Dimna in the Logging Industry

### Solution 1:  optimize dp with convex hull trick

Need observation that you will always cut trees that are at a greater index than current cut tree.  The reason for this is because you want to get to bn = 0 as quick, cause then it is free to recharge and cost 0 to cut trees. 

This observation is the tough part about this problem though, if I cut ith tree, why would I never cut (i-1)th tree but also tree j, j > i.  The reason this makes sense is because if you cut (i - 1th) tree with the current cost, it will cost some amount, but you had to cut a taller tree with a greater multiplier.  Why would you then go back with this smaller multplier.  it doesn't make sense for instance.

this one example shows that maybe you'd want to jump backwards and cut a tree that was before.  But hmmm is that better, or is it better to just continue to get way to bn = 0, then it will be free. 
a = 1 2 3 4 5
b = 5 4 3 2 0
first I cut tree a1 = 1, then maybe I cut tree a2 = 2 then tree a3 = 3
cost = b1*a2 + b2*a3 = 5*2 + 4*3 = 22
but if I cut tree a1 = 1, then tree a3 = 3, then tree a2 = 2
cost = b1*a3 + b3*a2 = 5*3 + 3*2 = 21 (but this doesn't help cause still going to cost 3 multiplier for next one, so yeah doesn't make sense to go backwards until you get bn = 0) (yeah so going backwards is waste of cost, cause you will have to still use that 3 to get closer to the bn = 0. )

```cpp
const long long is_query = -(1LL<<62);
struct line {
    long long m, b;
    mutable function<const line*()> succ;
    bool operator<(const line& rhs) const {
        if (rhs.b != is_query) return m < rhs.m;
        const line* s = succ();
        if (!s) return 0;
        long long x = rhs.m;
        return b - s->b < (s->m - m) * x;
    }
};

struct dynamic_hull : public multiset<line> { // will maintain upper hull for maximum
    const long long inf = LLONG_MAX;
    bool bad(iterator y) {
        auto z = next(y);
        if (y == begin()) {
            if (z == end()) return 0;
            return y->m == z->m && y->b <= z->b;
        }
        auto x = prev(y);
        if (z == end()) return y->m == x->m && y->b <= x->b;

		/* compare two lines by slope, make sure denominator is not 0 */
        long long v1 = (x->b - y->b);
        if (y->m == x->m) v1 = x->b > y->b ? inf : -inf;
        else v1 /= (y->m - x->m);
        long long v2 = (y->b - z->b);
        if (z->m == y->m) v2 = y->b > z->b ? inf : -inf;
        else v2 /= (z->m - y->m);
        return v1 >= v2;
    }
    void insert_line(long long m, long long b) {
        auto y = insert({ m, b });
        y->succ = [=] { return next(y) == end() ? 0 : &*next(y); };
        if (bad(y)) { erase(y); return; }
        while (next(y) != end() && bad(next(y))) erase(next(y));
        while (y != begin() && bad(prev(y))) erase(prev(y));
    }
    long long eval(long long x) {
        auto l = *lower_bound((line) { x, is_query });
        return l.m * x + l.b;
    }
};

int main(){
	int n = read();
	vector<long long> a(n), b(n), dp(n);
	for (int i = 0; i < n; ++i) {
		a[i] = read();
	}
	for (int i = 0; i < n; ++i) b[i] = read();
	dynamic_hull cht;
	cht.insert_line(-b[0], 0);
	for (int i = 1; i < n; i++) {
		// eval for x value
		dp[i] = -cht.eval(a[i]);
		// insert (m, b) line
		// for minimize take - (m, b), negative of slope and y-intercept
		cht.insert_line(-b[i], -dp[i]);
	}
	cout << dp.end()[-1] << endl;
}
```

## F. Pencils and Boxes

### Solution 1: 

```cpp

```

## G. Trains and Statistic

### Solution 1:  min segment tree + dp with segment tree optimization

```py
import math

class SegmentTree:
    def __init__(self, n: int, neutral: int, func):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)]

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx])
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.nodes[segment_idx] = val
        self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.func(result, self.nodes[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"nodes array: {self.nodes}, next array: {self.nodes}"
    
def main():
    n = int(input())
    arr = list(map(lambda x: int(x) - 1, input().split()))
    st = SegmentTree(n, math.inf, min)
    dp = [0]*n
    st.update(n - 1, n - 1)
    for i in range(n - 2, -1, -1):
        dp[i] = n - 1 - arr[i] - i + st.query(i + 1, arr[i] + 1)
        st.update(i, dp[i] + i)
    return sum(dp)

if __name__ == '__main__':
    print(main())
```

## H. The Fair Nut and Rectangles

### Solution 1:  convex hull trick for dp + dp recurrence + ad hoc

```cpp
const long long is_query = -(1LL<<62);
struct line {
    long long m, b;
    mutable function<const line*()> succ;
    bool operator<(const line& rhs) const {
        if (rhs.b != is_query) return m < rhs.m;
        const line* s = succ();
        if (!s) return 0;
        long long x = rhs.m;
        return b - s->b < (s->m - m) * x;
    }
};

struct dynamic_hull : public multiset<line> { // will maintain upper hull for maximum
    const long long inf = LLONG_MAX;
    bool bad(iterator y) {
        auto z = next(y);
        if (y == begin()) {
            if (z == end()) return 0;
            return y->m == z->m && y->b <= z->b;
        }
        auto x = prev(y);
        if (z == end()) return y->m == x->m && y->b <= x->b;

		/* compare two lines by slope, make sure denominator is not 0 */
        long long v1 = (x->b - y->b);
        if (y->m == x->m) v1 = x->b > y->b ? inf : -inf;
        else v1 /= (y->m - x->m);
        long long v2 = (y->b - z->b);
        if (z->m == y->m) v2 = y->b > z->b ? inf : -inf;
        else v2 /= (z->m - y->m);
        return v1 >= v2;
    }
    void insert_line(long long m, long long b) {
        auto y = insert({ m, b });
        y->succ = [=] { return next(y) == end() ? 0 : &*next(y); };
        if (bad(y)) { erase(y); return; }
        while (next(y) != end() && bad(next(y))) erase(next(y));
        while (y != begin() && bad(prev(y))) erase(prev(y));
    }
    long long eval(long long x) {
        auto l = *lower_bound((line) { x, is_query });
        return l.m * x + l.b;
    }
};

struct Point {
    long long x, y, a;
};

int main(){
	int n = read();
	vector<long long> dp(n + 1);
    vector<Point> points(n);
	for (int i = 0; i < n; ++i) {
        long long x = readll(), y = readll(), a = readll();
        points[i] = {x, y, a};
	}
    sort(points.begin(), points.end(), [](auto &a, auto &b) {return a.y > b.y;});
	dynamic_hull cht;
	cht.insert_line(0, 0);
	for (int i = 1; i < n + 1; i++) {
        long long x = points[i - 1].x, y = points[i - 1].y, a = points[i - 1].a;
		// eval for x value
		dp[i] = x*y - a + cht.eval(y);
		// insert (m, b) line
		// for minimize take - (m, b), negative of slope and y-intercept
		cht.insert_line(-x, dp[i]);
	}
	auto mx = max_element(dp.begin(), dp.end());
	cout << *mx << endl;
}
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

