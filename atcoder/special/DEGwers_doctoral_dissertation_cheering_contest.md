# DEGwer's Doctoral Dissertation Cheering Contest

## A - DEGwer's Doctoral Dissertation 

### Solution 1:  sliding window + deque

```py
from collections import deque

def main():
    N, K, T = map(int, input().split())
    arr = sorted(map(int, input().split()))
    res = 0
    queue = deque()
    for num in arr:
        queue.append(num)
        while queue and num - queue[0] >= T: # remove elements outside of current range
            queue.popleft()
        while len(queue) > 1: # remove the multiple typos
            queue.pop()
            res += 1
    print(res)

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```
