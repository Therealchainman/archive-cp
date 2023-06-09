# digit dp

Dynamic Programming on digits are one of the easier problems to solve with dynammic programming.  The example will be solving here is going to be Classy Numbers from Codeforces.  These problems usually involved being given a range of integers [L, R], where the range of integers can be very large such as 10^18, making it infeasible to iterate through.  Instead you can use dynamic programming on the digits to solve these. And solve for dp(R) - dp(L - 1) to compute the result in the range [L, R].  One variable that is common in these is you need to track `tight`, this is a boolean field in the dp states that represents if the prefix is equal to the prefix of the number you are solving for.  Such as if dp(1234), if you are at the current state of 12, then you match the prefix of 12, so it would be in a state of dp(i, tight), where i = index in the digits, so i = 2, cause we are at the 3rd digit, and tight = True because the prefix is 12, so that means for the next iteration you can only consider the digits [0,3].

When tight = False you can consider digits [0,9], because it doesn't matter you will never exceed the number solving for. 

Let's call some positive integer classy if its decimal representation contains no more than 3 non-zero digits


## Classy Numbers

### Solution 1:  digit dp

dp(i, j, tight) = count of classy numbers consider i digits, j nonzero digits, and tight bound. 

```py
from itertools import product

def count(n):
    digits = str(n)
    num_digits = len(digits)
    # dp(i, j, t), ith index in digits, j nonzero digits, t represents tight bound
    dp = [[[0]* 2 for _ in range(num_digits + 1)] for _ in range(num_digits + 1)]
    for i in range(int(digits[0]) + 1):
        dp[1][1 if i > 0 else 0][1 if i == int(digits[0]) else 0] += 1
    for i, t in product(range(1, num_digits), range(2)):
        for j in range(i + 1):
            for k in range(10): # digits
                if t and k > int(digits[i]): break
                dp[i + 1][j + (1 if k > 0 else 0)][t and k == int(digits[i])] += dp[i][j][t]
    return sum(dp[num_digits][j][t] for j, t in product(range(min(num_digits, 3) + 1), range(2)))

def main():
    left, right = map(int, input().split())
    res = count(right) - count(left - 1)
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```