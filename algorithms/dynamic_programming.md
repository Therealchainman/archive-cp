# Dynamic Programming

## Interval dynamic programming with cyclic dependency

To handle this interval, you can modify the intervals knowing it wouldn't affect the final answer.  interval dynamic programming where dp1 depends on dp2 and dp2 depends on dp1.  In a sense you have two different precomputations that are depending on each other.

```py
sys.setrecursionlimit(10**6)
import math
UNVISITED = -1

def main():
    n, x = map(int, input().split())
    arr = list(map(int, input().split()))
    dp1 = [[[UNVISITED] * (x + 1) for _ in range(n)] for _ in range(n)]
    dp2 = [[[UNVISITED] * (x + 1) for _ in range(n)] for _ in range(n)]
    nxt1, prv1 = [[0] * (x + 1) for _ in range(n)], [[0] * (x + 1) for _ in range(n)]
    nxt2, prv2 = [[0] * (x + 1) for _ in range(n)], [[0] * (x + 1) for _ in range(n)]
    for k in range(1, x + 1):
        last1 = last2 = n
        for i in reversed(range(n)):
            if arr[i] != k: last1 = i
            else: last2 = i
            nxt1[i][k] = last1
            nxt2[i][k] = last2
        first1 = first2 = -1
        for i in range(n):
            if arr[i] != k: first1 = i
            else: first2 = i
            prv1[i][k] = first1
            prv2[i][k] = first2
    # add all k
    def add(left, right, k):
        left = nxt1[left][k]
        right = prv1[right][k]
        if left > right: return 0
        if dp1[left][right][k] != UNVISITED: return dp1[left][right][k]
        res = math.inf
        # split
        for i in range(left, right):
            res = min(res, add(left, i, k) + add(i + 1, right, k))
        # transformation
        res = min(res, remove(left, right, k) + 1)
        dp1[left][right][k] = res
        return res
    # remove all k
    def remove(left, right, k):
        left = nxt2[left][k]
        right = prv2[right][k]
        if left > right: return 0
        if dp2[left][right][k] != UNVISITED: return dp2[left][right][k]
        res = math.inf
        # split
        for i in range(left, right):
            res = min(res, remove(left, i, k) + remove(i + 1, right, k))
        # transformation
        for m in range(1, x + 1):
            if m == k: continue
            res = min(res, add(left, right, m))
        dp2[left][right][k] = res
        return res
    ans = min([add(0, n - 1, k) for k in range(1, x + 1)])
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```