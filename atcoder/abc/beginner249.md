# Atcoder Beginner Contest 249

## Summary

index trio was a silly problem that I couldn't figure out either.  I imagine it may be a form of prime sieve algorithm because then I can find the possible divisibles for each integer
it will just be the primes, but of course I don't know that will work cause 

I need to understand run length encoding more.  Of course this problem is asking more about when does run length encoding result in a shorter string and I suppose lossless data compression? 
So you need some characters to repeat a certain number of times to reduce.  My observations from this were the following. 

Suppose you have x characters adjacent that are the same character
xxx => x3
xxxx => x4
xxxxx => x5
for length of 1 you are saving -1 character
for length of 2 you are saving 0 character
for length of 3 you are saving 1 character
for length of 4 you are saving 2 characters
for length of 5 you are saving 3 characters
Thus we can get the following mathematical equation

space_saved = length_of_identical_adjacent - 2
thus space is only saved for at least length of 3 or greater characters

Now to be honest with this equation for run length encoding, just a simple linear equation, it probably is not difficult to derive the number of ways to have efficient run length encoding. 

## Jogging

### Solution 1:  Simulation

```py
DRAW = "Draw"
TAK = "Takahashi"
AO = "Aoki"
 
def main():
    A, B, C, D, E, F, X = map(int, input().split())
    tak_dist = get_distance(A,B,C, X)
    ao_dist = get_distance(D,E,F, X)
    if tak_dist > ao_dist:
        return TAK
    if ao_dist > tak_dist:
        return AO
    return DRAW
 
def get_distance(walk, speed, rest, time):
    distance = 0
    while time > 0:
        travel = min(time, walk)
        distance += speed*travel
        time = time - travel - rest
    return distance
 
if __name__ == '__main__':
    print(main())
```

## Perfect String

### Solution 1: hash table + bitwise for boolean

```py
def main():
    S = input()
    seen = set()
    has_upper = has_lower = False
    for ch in S:
        if ch in seen: return False
        seen.add(ch)
        has_upper |= ch.isupper()
        has_lower |= ch.islower()
    return has_upper and has_lower
 
if __name__ == '__main__':
    if main():
        print("Yes")
    else:
        print("No")
```

## Just K

### Solution 1: Bit Masking 

```py
def main():
    N, K = map(int, input().split())
    S = [input() for _ in range(N)]
    best = 0
    for i in range(1, 1<<N):
        freq = [0]*26
        for j in range(N):
            if (i>>j)&1:
                for x in map(lambda x: ord(x)-ord('a'), S[j]):
                    freq[x] += 1
        best = max(best, sum(x==K for x in freq))
    return best
 
 
if __name__ == '__main__':
    print(main())
```

## Index Trio

### Solution 1:  Get all factors + simple combinatoric of number of ways when you have 2 distinct categories to choose from

O(nsqrt(n))

```py
from collections import Counter
from math import sqrt
def main():
    N = int(input())
    A = list(map(int,input().split()))
    counter = Counter(A)
    ways = 0
    for Ai in A:
        div_arr = divisors(Ai)
        for Aj in div_arr:
            Ak = Ai//Aj
            ways += counter.get(Ak, 0) * counter.get(Aj, 0)
    return ways

def divisors(num):
    div_arr = []
    for i in range(1, int(sqrt(num))+1):
        if num%i==0:
            div_arr.append(i)
            div_arr.append(num//i)
    return list(set(div_arr))
 
 
if __name__ == '__main__':
    print(main())
```

## RLE

### Solution 1: Dynamic Programming 

this is a hard one for me

```py

```

## Ignore Operations

### Solution 1: difference array 

I was wrong it looks greedy with heap being used, still don't really understand it. 

```py

```