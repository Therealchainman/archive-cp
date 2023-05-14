# Codeforces Round 872 Div 23

## Notes

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
from typing import *
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')
 
# Fast IO Region
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")
```

## A. LuoTianyi and the Palindrome String

### Solution 1:  greedy

```py
def main():
    s = input()
    if all(ch == s[0] for ch in s):
        return -1
    return len(s) - 1
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## B. LuoTianyi and the Table

### Solution 1:  greedy

figure out the best output and how many sub tables it will exist in. Take two conditions, either left top element is max or min value in the array.

```py
def main():
    n, m = map(int, input().split())
    arr = sorted(list(map(int, input().split())))
    if n < m:
        n, m = m, n
    res = 0
    max_diff1, max_diff2 = arr[-1] - arr[0], arr[-1] - arr[1]
    cur = (n*m - m) * max_diff1 + (m-1) * max_diff2
    res = max(res, cur)
    max_diff1, max_diff2 = arr[-1] - arr[0], arr[-2] - arr[0]
    cur = (n*m - m) * max_diff1 + (m-1) * max_diff2
    res = max(res, cur)
    return res
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## C. LuoTianyi and the Show

### Solution 1:  prefix and suffix count + pivot at each assigned seating position + sort

```py
def main():
    n, m = map(int, input().split())
    people = list(map(int, input().split()))
    seating = set()
    for x in people:
        if x > 0:
            seating.add(x)
    seating = sorted(seating)
    prefix = people.count(-1)
    suffix = people.count(-2) + len(seating)
    res = min(m, suffix) # considering the suffix
    for pos in seating:
        suffix -= 1
        cur = min(pos - 1, prefix) + min(m - pos, suffix) + 1
        prefix += 1
        res = max(res, cur)
    res = max(res, min(m, prefix))
    return res
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## D1. LuoTianyi and the Floating Islands (Easy Version)

### Solution 1:  tree dp

```py

```

## 

### Solution 1: 

```py

```

## E. LuoTianyi and XOR-Tree

### Solution 1:  tree dp

at most you can have 100,000 unique xor values for each root to leaf path. 

```py
def main():
    n = int(input())
    values = [0] + list(map(int, input().split()))
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    res = 0
    def dfs(node, parent, sum_):
        nonlocal res
        sum_ ^= values[node]
        if parent != -1 and len(adj_list[node]) == 1:
            # print('sum_', sum_, 'node', node)
            return sum_
        mask = (1 >> 20) - 1
        child_mask = 0
        freq = [0]*20
        for child in adj_list[node]:
            if child == parent: continue
            subtree_mask = dfs(child, node, sum_)
            print('subtree_mask', bin(subtree_mask), 'node', node, 'child', child)
            for i in range(20):
                if (subtree_mask>>i)&1:
                    freq[i] += 1
            child_mask |= subtree_mask
            mask &= subtree_mask
        print('node', node, 'mask', bin(mask), 'child_mask', bin(child_mask))
        child_mask ^= mask
        for i in range(20):
            if (child_mask>>i)&1:
                res += freq[i]
        print('res', res, 'node', node, 'mask', bin(mask), 'child_mask', bin(child_mask))
        return mask
    last_mask = dfs(1, -1, 0)
    res += bin(last_mask).count('1')
    return res

if __name__ == '__main__':
    # T = int(input())
    # for _ in range(T):
        # print(main())
    print(main())
```

This one overcounts on test case 8 some reason, but I can't determine what is incorrect about this approach to the problem. 

```cpp
const int N = 1e5 + 5;

vector<long long> values;
vector<vector<int>> adj_list;
unordered_map<long long, int> min_ops[N];

int is_leaf(int node, int parent) {
    return adj_list[node].size() == 1 && parent != -1;
}

int dfs(int node, int parent) {
    if (is_leaf(node, parent)) {
        min_ops[node][values[node]] = 0;
        return values[node];
    }
    unordered_map<long long, int> freq;
    int operation_count = 0;
    for (int child : adj_list[node]) {
        if (child == parent) continue;
        long long val = dfs(child, node);
        int ops = min_ops[child][val];
        freq[val]++;
        operation_count += ops;
    }
    // int maxer = max_element(freq.begin(), freq.end(), [](const pair<int, int>& x, const pair<int, int>& y){ return x.second < y.second; })->second;
    // int sum = accumulate(freq.begin(), freq.end(), 0, [](const int& a, const auto& b) { return a + b.second; });
    // printf("max_count: %d, sum: %d\n", maxer, sum);
    int total_cost = operation_count + accumulate(freq.begin(), freq.end(), 0, [](const int& a, const auto& b) { return a + b.second; }) - max_element(freq.begin(), freq.end(), [](const pair<int, int>& x, const pair<int, int>& y){ return x.second < y.second; })->second;
    long long max_value = max_element(freq.begin(), freq.end(), [](const pair<int, int>& x, const pair<int, int>& y){ return x.second < y.second; })->first;
    long long new_val = max_value ^ values[node];
    min_ops[node][new_val] = total_cost;
    // printf("node: %d, new_val: %lld, max_val: %lld, total_cost: %d\n", node, new_val, max_value, total_cost);
    min_ops[node][0] = total_cost + (new_val != 0);
    return new_val;
}

int main() {
    int n = read();
    values.resize(n + 1);
    adj_list.resize(n + 1);
    for (int i = 1; i <= n; i++) {
        // long long val = readll();
        values[i] = readll();
    }
    for (int i = 1; i < n; i++) {
        int u = read(), v = read();
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }
    dfs(1, -1);
    cout << min_ops[1][0] << endl;
    return 0;
}
```