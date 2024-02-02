# Constructor's Open Cup 2024

## A. Houses

### Solution 1:  math

```py
def main():
    x1, x2, x = map(int, input().split())
    if x1 > x2: x1, x2 = x2, x1
    print("YES" if x1 <= x <= x2 else "NO")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
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

## E. Dice Game

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```

## G. Coin Tossing

### Solution 1:  sliding window

```py
def main():
    n, k = map(int, input().split())
    s = input()
    l = wcount = 0
    ans = 1
    for r in range(1, n):
        if s[r] != s[r - 1]: wcount += 1
        while wcount > k:
            if s[l] != s[l + 1]: wcount -= 1
            l += 1
        ans = max(ans, r - l + 1)
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## H. GCD Set

### Solution 1:  primes, math

Using the idea if you multiple all the primes by x, than it is guaranteed that each integer is x * a different prime, and thus the gcd between any pair of integer is going to be x.

```py
PRIMES = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523]

def main():
    n, x = map(int, input().split())
    ans = [0] * n
    for i in range(n):
        ans[i] = x * PRIMES[i]
    print(*ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## I. Pairing Numbers

### Solution 1:  sort, greedy

pair up largest negative numbers, and then pair any remaining negatives with 0, or just leave one negative unpaired,  the only other thing than a negative to pair would be a 0 though.  
Than you can pair up the positive integers.  with the largest first, and than once a pair is less than the sum of the small numbers don't pair them okay. 



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

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```