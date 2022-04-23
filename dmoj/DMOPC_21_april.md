# DMOPC '21 April Contest

## Summary

Interesting problems, but I really couldn't come up with any good approaches for any of them.  I was able to only get points from problem 1, and it was full points by using a method of a queue + stack


## Peanut Planning

### Solution 1: queue + stack 

```py
from collections import deque
def main():
    N, M = map(int,input().split())
    A = sorted(list(map(int,input().split())))
    queue = deque(A)
    res = []
    stack = []
    while queue:
        first = queue.popleft()
        res.append(first)
        while queue and queue[-1] + first >= M:
            stack.append(queue.pop())
        if stack:
            second = stack.pop()
            res.append(second)
    while stack:
        res.append(stack.pop())
    for i in range(1,N):
        if res[i]+res[i-1] < M: return -1
    return " ".join(map(str, res))
if __name__ == '__main__':
    print(main())
```