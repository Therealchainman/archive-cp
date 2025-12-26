# Fenwick Trees

## SUMMARY

Fenwick trees are a data structure that can be used to efficiently calculate dynamic prefix sums in a table of numbers.

Popularly also known as Binary Indexed Trees (BIT).

### IMPLEMENTED IN C++ PURQ

The thing that I need to know about a fenwick tree datastructure is how to use it. It is useful for when you need to 
modify the range sum,  So with this you can both update a range sum in the tree, and query a range sum in log(n) time complexity

This equation is 1-indexed based, so that means it starts at index=1, so if you have start at index 0 need to add 1 to all the values

Initialize it with the following
self.fenwick = FenwickTree(n)
Do not need to do n + 1, it does that so it already works for 1-index arrays.

self.fenwick.update(r+1,-k)

if I query(5) it looks in the range [0,5], so it is inclusive

point update range queries

If you have an array of values, and you are updating a value from the array, you need to update the fenwick tree with the difference between the new value and the old value, so that the fenwick tree is updated with the new value.  It is all incremental.  So if you have a value of 5, and you update it to 10, you need to update the fenwick tree with 5, so that the fenwick tree is updated with the new value.

But sometimes fenwick tree is used for counting, or marking if something is currently selected, in those case you usually adding 1 or -1 to the fenwick tree.

```cpp
template <typename T>
struct FenwickTree {
    vector<T> nodes;
    T neutral;

    FenwickTree() : neutral(T(0)) {}

    void init(int n, T neutral_val = T(0)) {
        neutral = neutral_val;
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, T val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    T query(int idx) {
        T result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }

    T query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : T(0);
    }
};
```

Another version of the data structure with modular arithmetic

```cpp

template <typename T>
struct FenwickTree {
    vector<T> nodes;
    T neutral;

    FenwickTree() : neutral(T(0)) {}

    void init(int n, T neutral_val = T(0)) {
        neutral = neutral_val;
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, T val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] = (nodes[idx] + val) % MOD;    
            idx += (idx & -idx);
        }
    }

    T query(int idx) {
        T result = neutral;
        while (idx > 0) {
            result = (result + nodes[idx]) % MOD;
            idx -= (idx & -idx);
        }
        return result;
    }

    T query(int left, int right) {
        int ans = right >= left ? query(right) - query(left - 1) : T(0);
        if (ans < 0) ans += MOD;
        return ans;
    }
};
```

## 2D Fenwick Tree

The 2-dimensional fenwick tree for dynamic rectangular sum queries

supports point updates and rectangular sum queries in logarithmic time.

1-indexed just like above but for rectangular sum queries r1, c1, r2, c2

O(log^2(n)) for the queries

But to 0 fill the bit it is O(N^2)
It is inclusive for the rectangle sum query

```cpp
struct BIT2D {
    int n;
    vector<vector<int64>> bit;
    BIT2D(int n) : n(n), bit(n + 1, vector<int64>(n + 1, 0)) {}

    void add(int r, int c, int64 delta) {
        for (int i = r; i <= n; i += i & -i) {
            for (int j = c; j <= n; j += j & -j) {
                bit[i][j] += delta;
            }
        }
    }

    int64 sum(int r, int c) const {
        int64 res = 0;
        for (int i = r; i > 0; i -= i & -i) {
            for (int j = c; j > 0; j -= j & -j) {
                res += bit[i][j];
            }
        }
        return res;
    }

    int64 rect(int r1, int c1, int r2, int c2) const {
        if (r1 > r2) swap(r1, r2);
        if (c1 > c2) swap(c1, c2);
        int64 res = sum(r2, c2) - sum(r1 - 1, c2) - sum(r2, c1 - 1) + sum(r1 - 1, c1 - 1);
        return res;
    }
};
```

### IMPLEMENTED IN PYTHON + PURQ

The thing that I need to know about a fenwick tree datastructure is how to use it. It is useful for when you need to 
modify the range sum,  So with this you can both update a range sum in the tree, and query a range sum in log(n) time complexity

This equation is 1-indexed based, so that means it starts at index=1, so if you have start at index 0 need to add 1 to all the values

Initialize it with the following
self.fenwick = FenwickTree(n)

self.fenwick.update(r+1,-k)

if I query(5) it looks in the range [0,5], so it is inclusive

point update range queries

```py
class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def query_range(self, i, j):
        return self.query(j) - self.query(i - 1) if j >= i else 0

    def __repr__(self):
        return f"array: {self.sums}"
```

How it works for queries and updates,  you cannot do update(0, x), because it causes it to run infinite loop, but you can query from 0, but it is not necessary since verything needs to be treated as 1-indexed

```py
fenwick = FenwickTree(MAXN)
fenwick.update(1, 1)
fenwick.update(2, 1)
print(fenwick)
assert fenwick.query_range(0, 0) == 0, "Wrong"
assert fenwick.query_range(0, 1) == 1, "Wrong"
assert fenwick.query_range(0, 2) == 2, "Wrong"
assert fenwick.query_range(1, 2) == 2, "Wrong"
assert fenwick.query_range(2, 2) == 1, "Wrong"
```

Number of inversions is easy to calculate with Fenwick Trees
