# INTERACTIVE PROBLEMS

Example for problem, flush may not be necessary, it worked without, any thing above doesn't matter, can keep normal template for fast i/o

Surprisingly there is not anything extra needed for interactive problems in codeforces. 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    psum = [0]*(n + 1)
    for i in range(n):
        psum[i+1] = psum[i] + arr[i]
    left, right = 0, n - 1
    while left < right:
        mid = (left + right) >> 1
        size = mid - left + 1
        print('?', size, *range(left + 1, mid + 2), flush = True)
        x = int(input())
        if x > psum[mid + 1] - psum[left]:
            right = mid
        else:
            left = mid + 1
    print('!', left + 1, flush = True)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```