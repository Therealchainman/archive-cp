# Google Code Jam 2022

# Google Codejam Round 1B

## Pancake Deque

### Solution 1: double ended queue + greedily choose the smallest pancake

```py
from collections import deque
def main():
    N = int(input())
    D = deque(map(int,input().split()))
    prev = cnt = 0
    while D:
        if D[0] < D[-1]:
            cost = D.popleft()
        else:
            cost = D.pop()
        if cost >= prev:
            cnt += 1
        prev = max(prev, cost)
    return cnt
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Controlled Inflation

### Solution 1: dynamic programming with the max and mins + greedy

This assumes that the best way is always to travel from either min or max of previous array
and then travel from there across the current array and end at min or max

```py
from functools import lru_cache
import sys
sys.setrecursionlimit(1000000)
def main():
    N, P = map(int,input().split())
    products = [list(map(int,input().split())) for _ in range(N)]
    mins, maxs = [min(products[i]) for i in range(N)], [max(products[i]) for i in range(N)]
    
    @lru_cache(None)
    def dfs(i, last_min):
        if i==N: return 0
        dist = maxs[i] - mins[i]
        prev = 0 if i==0 else mins[i-1] if last_min else maxs[i-1]
        return dist + min(
            abs(prev-mins[i]) + dfs(i+1, False),
            abs(prev-maxs[i]) + dfs(i+1, True)
        )
    return dfs(0, True)
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## ASeDatAb

### Solution 1: Simply pick so that eventually it will become 11111111

Only passes test set 1

```py
from random import shuffle
import sys
def main():
    BINARY = '10000000'
    while True:
        print(BINARY, flush=True)
        N = int(input())
        if N == 0:
            return True
        elif N == -1:
            return False
        arr = []
        for _ in range(N):
            arr.append('1')
        for _ in range(8-N):
            arr.append('0')
        shuffle(arr)
        BINARY = "".join(arr)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

# Google Codejam Round 1C

## Letter Blocks

### Solution 1: greedy + break it down into starting characters and extending characters

```py
from collections import defaultdict
import string
IMP = "IMPOSSIBLE"
def main():
    N = int(input())
    prefix, suffix, single = defaultdict(list), defaultdict(list), defaultdict(list)
    middle = defaultdict(list)
    S = input().split()
    for i, s in enumerate(S):
        seen = [0]*26
        prev = s[0]
        start, end = s[0], s[-1]
        for ch in s:
            x = ord(ch)-ord('A')
            if ch != prev and seen[x]: return IMP
            if ch not in (prev, start, end):
                middle[ch].append(i)
            seen[x] = 1
            prev = ch
        # ADD ELEMENT TO THE SINGLES IF START EQUAL ENDS
        if start == end:
            single[start].append(i)
        else: # ADD ELEMENT TO PREFIX AND SUFFIX
            prefix[start].append(i)
            suffix[end].append(i)
    # CANDIDATE LIST FOR CHARACTERS YOU CAN START WITH
    candidates = []
    # REST CAN ONLY BE USED AS EXTENSION BLOCKS
    # IMPOSSIBLE CONDITIONS ARE 
    for ch in string.ascii_uppercase:
        MIDDLE_LEN, PREFIX_LEN, SUFFIX_LEN, SINGLE_LEN = len(middle[ch]), len(prefix[ch]), len(suffix[ch]), len(single[ch])
        if MIDDLE_LEN > 1 or PREFIX_LEN > 1 or SUFFIX_LEN > 1: return IMP
        if MIDDLE_LEN == 1 and (PREFIX_LEN > 0 or SUFFIX_LEN > 0 or SINGLE_LEN > 0): return IMP
        if MIDDLE_LEN == 1 or SUFFIX_LEN == 1: continue
        if MIDDLE_LEN==PREFIX_LEN==SUFFIX_LEN==SINGLE_LEN==0: continue
        candidates.append(ch)
    # NOW WE WANT TO MOVE THROUGH CANDIDATES UNTIL THE END
    result = []
    current_char = -1
    while candidates or current_char != -1:
        if current_char == -1: # NEED STARTING CHARACTER
            current_char = candidates.pop()
            for index in single[current_char]:
                result.append(index)
            for index in prefix[current_char]:
                result.append(index)
        else: # NEED EXTENDING CHARACTER
            for index in single[current_char]:
                result.append(index)
            for index in prefix[current_char]:
                result.append(index)
        next_char = S[result[-1]][-1]
        if next_char == current_char:
            current_char = -1
        else:
            current_char = next_char
        
        
    if len(result) != len(S): return IMP
    return "".join([S[i] for i in result])
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Squary

### Solution 1: math + finding out need the sum of distinct multiplicative pairs to be equal to 0

![proof](squary.PNG)

![proof2](squary2.PNG)

```py
from itertools import product
IMP = "IMPOSSIBLE"
def main():
    N, K = map(int,input().split())
    arr = list(map(int,input().split()))
    SP = (sum(x*y for x, y in product(arr,repeat=2)) - sum(x*x for x in arr))//2
    S = sum(arr)
    if K == 1:
        if SP==S==0: return 1
        if S==0: return IMP
        return -SP//S if SP%S==0 else IMP
    n1 = 1-S
    n2 = -(SP+n1*S)
    return f"{n1} {n2}"
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Intranets

Need to know how to take the multiplicative modular inverse

```py

```

# Google Codejam Round 2

## Pixelated Circle

### Solution 1:  Solve test case 1 with set theory, brute force to compute the circles in sets, then find the outer join of the sets, basically find all elements that are not in both sets. 

```py
from math import sqrt
from itertools import product
def main():
    row = int(input())
    fill_circle = set()
    fill_circle_wrong = set()
    def draw_circle_filled_wrong(R):
        for r in range(R+1):
            draw_circle_perimeter(r)
    def draw_circle_perimeter(R):
        for x in range(-R, R+1):
            y = round(sqrt(R*R-x*x))
            fill_circle_wrong.add((x,y))
            fill_circle_wrong.add((x,-y))
            fill_circle_wrong.add((y,x))
            fill_circle_wrong.add((-y,x))
    def draw_circle_filled(R):
        for x, y in product(range(-R,R+1), repeat=2):
            if round(sqrt(x*x+y*y)) <= R:
                fill_circle.add((x,y))
    draw_circle_filled(row)
    draw_circle_filled_wrong(row)
    return len(fill_circle^fill_circle_wrong)
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```