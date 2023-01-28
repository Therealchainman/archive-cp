# Fenwick Trees

## SUMMARY

Fenwick trees are a data structure that can be used to efficiently calculate prefix sums in a table of numbers.

### IMPLEMENTED IN C++ 

This fenwick tree will provide super fast range sum queries
works for point updates and range queries

when updating will probably need to do ft.update(i + 1, 1)

if want to use it for counting, when object is added do ft.update(i + 1, 1) 
and when object is removed do ft.update(i + 1, -1)

range queries this will work ft.query(left - 1, right)

```cpp
int neutral = 0;
struct FenwickTree {
    vector<int> nodes;
    
    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, int val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int query(int left, int right) {
        return query(right) - query(left);
    }

    int query(int idx) {
        int result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }
};
```

### IMPLEMENTED IN PYTHON

The thing that I need to know about a fenwick tree datastructure is how to use it. It is useful for when you need to 
modify the range sum,  So with this you can both update a range sum in the tree, and query a range sum in log(n) time complexity

This equation is 1-indexed based, so that means it starts at index=1, so if you have start at index 0 need to add 1 to all the values

Initialize it with the following
self.fenwick = FenwickTree(n)

self.fenwick.update(r+1,-k)

if I query(5) it looks in the range [0,5], so it is inclusive

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

    def __repr__(self):
        return f"array: {self.sums}"
```