# Codeforces Round 836 Div 2

## A. SSeeeeiinngg DDoouubbllee

```py
def main():
    s = input()
    n = len(s)
    arr = ['a']*(2*n)
    for i, ch in enumerate(s):
        arr[i] = arr[~i] = ch
    return ''.join(arr)

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        print(main())
```

## B. XOR = Average

### Solution 1: math + arithmetic progression(sequence) + O(n) for each test case

```py
def main():
    n = int(input())
    if n&1:
        return ' '.join(map(str, [1]*n))
    return ' '.join(map(str, [1, 3] + [2]*(n-2)))


if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        print(main())
```