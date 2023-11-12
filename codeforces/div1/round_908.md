# Codeforces Round 908 div1

## B. Neutral Tonality

```py

```

## C. Freedom of Choice

```py

```

## D. Colorful Constructive

### max heap, frequency, greedy

```py
import heapq

def main():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    s = list(map(int, input().split()))
    d = list(map(int, input().split()))
    cnt = [0] * (n + 1)
    maxheap = []
    for num in a:
        cnt[num] += 1
    for i, c in enumerate(cnt):
        if c == 0: continue
        heapq.heappush(maxheap, (-c, i))
    ans = [[0] * s[i] for i in range(m)]
    for i in range(m):
        for j in range(s[i]):
            if j >= d[i] and cnt[ans[i][j - d[i]]] > 0:
                heapq.heappush(maxheap, (-cnt[ans[i][j - d[i]]], ans[i][j - d[i]]))
            if len(maxheap) == 0: return print(-1)
            _, num = heapq.heappop(maxheap)
            ans[i][j] = num
            cnt[num] -= 1
        for j in range(s[i], s[i] + d[i]):
            if cnt[ans[i][j - d[i]]] > 0:
                heapq.heappush(maxheap, (-cnt[ans[i][j - d[i]]], ans[i][j - d[i]]))
    for i in range(m):
        print(*ans[i])
                
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```