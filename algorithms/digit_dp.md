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

## Only zeros

For some problems you need to track if you have seen a nonzero digit up to this point.  

## Digit Sum Divisible

For some problems you need to track if something is divisible by an integer value.  So instead of looping over every integer, you can us dynamic programming, where you keep track of the remainder modulo n.  So that can be anything.  For one problem it could be the digit sum. 

Instead of digit sum, you can also track if an integer is divisble by k.  Using the fact that (x1%k+x2%k)%k = (x1+x2)%k.  So you can just track the value of everything modulo k throughout the entire problem, and know if the remainder is 0 or not for an integer.  You will need to have calculated powers array of (10^x) % k to perform operation. 

Here is an example of this one with digit dp.

```cpp
class Solution {
public:
    vector<vector<int>> dp;
    vector<vector<int>> track;
    vector<int> powers;
    int N, K, mid;
    int recurse(int i, int rem) {
        if (i > mid) return rem == 0;
        if (dp[i][rem] != -1) return dp[i][rem];
        for (int dig = 9; dig >= 0; dig--) {
            int add = (dig * powers[i]) % K;
            if (i < mid || (i == mid && N % 2 == 0)) add = (add + dig * powers[N - i - 1]) % K;
            if (recurse(i + 1, (rem + add) % K)) {
                track[i][rem] = dig;
                return dp[i][rem] = true;
            }
        }
        return dp[i][rem] = false;
    }
    string largestPalindrome(int n, int k) {
        N = n;
        K = k;
        mid = (N - 1) / 2;
        dp.assign(mid + 1, vector<int>(K, -1));
        track.assign(mid + 1, vector<int>(K, -1));
        powers.resize(N);
        powers[0] = 1;
        for (int i = 1; i < N; i++) {
            powers[i] = (powers[i - 1] * 10) % K;
        }
        recurse(0, 0);
        string ans = "";
        for (int i = 0, rem = 0; i <= mid; i++) {
            ans += track[i][rem] + '0';
            int dig = track[i][rem];
            int add = (dig * powers[i]) % K;
            if (i < mid || (i == mid && N % 2 == 0)) add = (add + dig * powers[N - i - 1]) % K;
            rem = (rem + add) % K;
        }
        string s = ans.substr(0, N / 2);
        reverse(s.begin(), s.end());
        ans += s;
        return ans;
    }
};
```
