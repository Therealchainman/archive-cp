# Codeforces Round 850 Div 2

## Notes

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
from typing import *
 
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

## A1. Non-alternating Deck (easy version)

### Solution 1:  math + modular arithmetic by 4 so get two consecutive turns

```py
def main():
    n = int(input())
    alice = bob = round = cards = 0
    while n > 0:
        if round%4 < 2:
            alice += min(n, cards)
        else:
            bob += min(n, cards)
        n -= cards
        cards += 1
        round += 1
    return ' '.join(map(str, [alice, bob]))
 
if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        print(main())
```

## A2. Alternating Deck (hard version)

### Solution 1: math + mod

```py
def main():
    n = int(input())
    alice_w = bob_w = alice_b = bob_b = round = left = cards = 0
    while n > 0:
        take = min(n, cards)
        right = left + take - 1 # inclusive
        if round%4 < 2:
            if left&1:
                alice_b += 1
            if right%2 == 0:
                alice_w += 1
                right -= 1
            num_cards = right - left + 1
            alice_w += num_cards//2
            alice_b += num_cards//2
        else:
            if left&1:
                bob_b += 1
            if right%2 == 0:
                bob_w += 1
                right -= 1
            num_cards = right - left + 1
            bob_w += num_cards//2
            bob_b += num_cards//2
        n -= take
        left += take
        cards += 1
        round += 1
    return ' '.join(map(str, [alice_w, alice_b, bob_w, bob_b]))

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        print(main())
```

## B. Cake Assembly Line

### Solution 1:  transformation of 1 dimensional coordinates + assign each dispenser to a cake + keep a range of possible positions for each dispenser like a window + update the window at based on the separation between dispensers and cakes 

```py
def main():
    n, w, h = map(int, input().split())
    cakes_cp = list(map(int, input().split()))
    dispensers_cp = list(map(int, input().split()))
    dist = 2*w - 2*h
    left = prev_left = h
    right = prev_right = h + dist
    shift_to_right = cakes_cp[0] - w
    for i in range(1, n):
        delta_disp = dispensers_cp[i] - dispensers_cp[i-1]
        left += delta_disp
        right += delta_disp
        delta_cake = cakes_cp[i] - cakes_cp[i-1]
        prev_left += delta_cake
        prev_right += delta_cake
        left = max(left, prev_left)
        right = min(right, prev_right)
        if left > right: return 'NO'
    return 'YES'



if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        print(main())
```

## C. Monsters (easy version)

### Solution 1:  bucket sort + greedy + loop

```py
def bucket_sort(nums: List[int]) -> List[int]:
    m = max(nums)
    bucket = [0] * (m + 1)
    for num in nums:
        bucket[num] += 1
    return bucket
 
def main():
    n = int(input())
    hp = list(map(int, input().split()))
    buckets = bucket_sort(hp)
    m = len(buckets)
    left, res, right = 1, 0, 0
    while right < m:
        while left < right and buckets[left] > 0:
            left += 1
        if buckets[right] > 0 and buckets[left] == 0:
            buckets[right] -= 1
            buckets[left] += 1
            res = res + (right - left)
        while (right < m and buckets[right] == 0) or right == left:
            right += 1
    return res
 
if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        print(main())
```

### Solution 2: math + sum of consecutive natural numbers sequence

```py

```

## D. Letter Exchange

### Solution 1:

```py

```

## E. Monsters (hard version)

### Solution 1: sorted list + binary search + storing used and extra values + reversed iteration + sum of natural numbers sequence

```py
def main():
    n = int(input())
    hp = list(map(int, input().split()))
    used_sum = 0
    used, extra = SortedList(), SortedList()
    for h in sorted(hp):
        if h > len(used):
            used.add(h)
            used_sum += h
        else:
            extra.add(h)
    arith_seq_sum = lambda x: x*(x+1)//2
    result = [0]*n
    for i in reversed(range(n)):
        result[i] = used_sum - arith_seq_sum(len(used))
        ei = extra.bisect_left(hp[i])
        ui = used.bisect_left(hp[i])
        # IF AN EXTRA IS MORE THAN OR EQUAL TO CURRENT HP, YOU WILL WANT TO MOVE THAT EXTRA INTO USED AND UPDATED THE SUM OF USED HP
        if ei < len(extra):
            used.add(extra[ei])
            used_sum += extra.pop(ei)
        # IF HP IS IN USED REMOVE IT AND UPDATE
        if ui < len(used):
            ui = used.bisect_left(hp[i])
            used.pop(ui)
            used_sum -= hp[i]
    return ' '.join(map(str, result))

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        print(main())
```