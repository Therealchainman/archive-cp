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

Leading zeros are special in counting problems. The zero flag tracks whether the number has actually started.

Typical reasons to keep this flag:
- To let numbers have shorter “effective” length than arr.size(). By placing leading zeros while zero = 1, you model numbers with fewer digits without counting different zero pads as different numbers.

Example: counting numbers with nondecreasing digits.

It counts the number of base-base numbers that are:
- not greater than an upper bound given as a digit array arr
- nondecreasing from left to right
- You enforce nondecreasing by remembering the last chosen digit and only allowing the next digit to be at least that value.

```cpp
int dfs(const vector<int>& arr, int i, int d, int tight, int zero) {
    if (i == arr.size()) return 1;
    if (dp[d][tight][zero][i] != -1) return dp[d][tight][zero][i];
    int ans = 0;
    for (int dig = d; dig < base; ++dig) {
        if (tight && dig > arr[i]) break;
        int term = dfs(arr, i + 1, dig, tight & (arr[i] == dig), zero & (dig == 0));
        ans = (ans + term) % MOD;
    }
    return dp[d][tight][zero][i] = ans;
}
```

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

## Example of digit dp

This example handles a scenario where you are counting the number of scenarios where the head digit is greater than all other digits.
This handles the case of the integers that have fewer number of digits than the integer x in a way that I think is easy to code and understand.  It just resets the ndp[i][i][0] = 1 in each layer.  

```cpp
int L, R;

int calc(int x) {
    // head, last, is tight, is zero
    vector<vector<vector<int>>> dp(10, vector<vector<int>>(10, vector<int>(2, 0)));
    string num = to_string(x);
    for (int i = 1; i < 10; i++) {
        int d = num[0] - '0';
        if (i > d) break;
        dp[i][i][i == d] = 1;
    }
    for (int idx = 1; idx < num.size(); idx++) {
        int d = num[idx] - '0';
        vector<vector<vector<int>>> ndp(10, vector<vector<int>>(10, vector<int>(2, 0)));        
        for (int i = 0; i < 10; i++) {
            if (i > 0) ndp[i][i][0] = 1;
            for (int j = 0; j <= i; j++) {
                for (int k = 0; k < 2; k++) {
                    for (int cur = 0; cur < 10; cur++) {
                        if (cur >= i) break;
                        if (k == 1 && cur > d) break;
                        ndp[i][cur][k == 1 && cur == d] += dp[i][j][k];
                    }
                }
            }
        }
        swap(dp, ndp);
    }
    int ans = 0;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 2; k++) {
                ans += dp[i][j][k];
            }
        }
    }
    return ans;
}

void solve() {
    cin >> L >> R;
    int ans = calc(R) - calc(L - 1);
    cout << ans << endl;
}
```
