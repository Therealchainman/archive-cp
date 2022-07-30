# A Math Contest

The only solution below only passes 6/11 cases, there must be some edge case it is not handling

# P15: Matrix Fixed Point

## Solution: math + gaussian elimination + linear algebra + solve a system of linear equations

```py
def invmod(elem, mod):
    return pow(elem, -1, mod)
def main():
    # INPUTS
    mod = int(1e9)+7
    n = int(input())
    a = []
    inp = []
    for _ in range(n):
        inp.append(list(map(int,input().split())))
    # PREPROCESS FOR GAUSS ELIMINATION
    for i in range(n-1):
        a.append(inp[i] + [0])
        a[i][i] = (a[i][i] - 1 + mod)%mod
    a.append([1]*(n+1))
    x = [0]*n
    # GAUSS ELIMINATION
    for i in range(n):
        for j in range(i+1,n):
            ratio = (a[j][i]*invmod(a[i][i],mod))%mod
            for k in range(n+1):
                a[j][k] = (a[j][k] - ratio*a[i][k] + mod)%mod
    # BACK SUBSTITUTION
    x[n-1] = (a[n-1][n]*invmod(a[n-1][n-1],mod))%mod
    # place results
    for i in range(n-2,-1,-1):
        x[i] = a[i][n]
        for j in range(i+1,n):
            x[i] = (x[i] - a[i][j]*x[j] + mod)%mod
        x[i] = (x[i]*invmod(a[i][i],mod))%mod
    return ' '.join(map(str, x))

if __name__ == '__main__':
    print(main())
```