# Dynamic Programming

## At the top of each script

```cpp
#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}
```

## Coin Combinations I

### Solution 1:  iterative dp + order doesn't matter + Unordered Coin Change + O(nx) time

This can be solved by having two for loops in a particular order.

Iterate through the sum of the coins first and then through the coins, and add the coins for that sum. This leads to adding up very quicly for example if you have

```py
coins = [2, 3, 5], x = 9
dp = [[], [], [], [], [], [], [], [], [], []]
for when coin_sum = 2 it becomes
dp = [[], [], [2], [], [], [], [], [], [], []]
coin_sum = 3
dp = [[], [], [2], [3], [], [], [], [], [], []]
coin_sum = 4
dp = [[], [], [2], [3], [2, 2], [], [], [], [], []]
coin_sum = 5
dp = [[], [], [2], [3], [2, 2], [[3, 2], [2, 3], [5]], [], [], [], []]
coin_sum = 6
dp = [[], [], [2], [3], [2, 2], [[3, 2], [2, 3], [5]], [[2, 2, 2], [3, 3]], [[3, 2, 2], [2, 3, 2], [5, 2], [2, 2, 3], [2, 5]], [], []]
```

```cpp
int main() {
    int n = read(), x = read();
    int mod = 1e9 + 7;
    vector<int> dp(x + 1, 0);
    dp[0] = 1;
    vector<int> coins;
    for (int i = 0; i < n; i++) {
        int c = read();
        coins.push_back(c);
    }
    for (int coin_sum = 1; coin_sum <= x; coin_sum++) {
        for (int i = 0; i < n; i++) {
            if (coins[i] > coin_sum) continue;
            dp[coin_sum] = (dp[coin_sum] + dp[coin_sum - coins[i]]) % mod;
        }
    }
    cout << dp[x] << endl;
    return 0;
}
```

## Coin Combinations II

### Solution 1: iterative dp + order matters + O(nx) time

For this problem you want to iterate through the coins first and then the coin_sum. This is because you want to add the coins in a particular order. For example if you have

```py
coins = [2, 3, 5], x = 9
dp = [[], [], [], [], [], [], [], [], [], []]
coin = 2
dp = [[], [], [2], [], [[2, 2]], [], [[2, 2, 2]], [], [[2, 2, 2, 2]], []]
coin = 3
dp = [[], [], [2], [[3]], [[2, 2]], [[2, 3]], [[2, 2, 2], [3, 3]], [[2, 2, 3]], [[2, 2, 2, 2], [2, 3, 3]], [[2, 2, 2, 3], [3, 3, 3]]]
coin = 5
dp = [[], [], [2], [[3]], [[2, 2]], [[2, 3], [5]], [[2, 2, 2], [3, 3]], [[2, 2, 3], [2, 5]], [[2, 2, 2, 2], [2, 3, 3], [3, 5]], [[2, 2, 2, 3], [3, 3, 3], [2, 2, 5]]]
```

```cpp
int main() {
    int n = read(), x = read();
    int mod = 1e9 + 7;
    vector<int> dp(x + 1, 0);
    dp[0] = 1;
    vector<int> coins;
    for (int i = 0; i < n; i++) {
        int c = read();
        coins.push_back(c);
    }
    for (int i = 0; i < n; i++) {
        for (int coin_sum = coins[i]; coin_sum <= x; coin_sum++) {
            if (coins[i] > coin_sum) continue;
            dp[coin_sum] = (dp[coin_sum] + dp[coin_sum - coins[i]]) % mod;
        }
    }
    cout << dp[x] << endl;
    return 0;
}
```

##

### Solution 1: 

```py

```

## Counting Towers

### Solution 1: 

```py
def main():
    n = int(input())
    mod = int(1e9) + 7
    psum = 1
    dp = 1
    for i in range(1, n + 1):
        psum += pow(2, 2 * i - 2, mod)
        psum %= mod
        print('psum', psum)
        dp = psum
        psum += dp
        psum %= mod
        # print(i, dp)
    return dp


if __name__ == '__main__':
    # print(main())
    T = int(input())
    for _ in range(T):
        print(main())
```

## Projects

### Solution 1:  sort + iterative dynammic programming + coordinates compression

```py
def main():
    n = int(input())
    events = []
    days = set()
    for i in range(n):
        a, b, p = map(int, input().split())
        events.append((a, -p, 0))
        events.append((b, p, a))
        days.update([a, b])
    compressed = {x: i + 1 for i, x in enumerate(sorted(days))}
    events.sort()
    dp = [0] * (len(compressed) + 1)
    for day, p, start in events:
        i = compressed[day]
        if p < 0:
            dp[i] = max(dp[i], dp[i - 1])
        else:
            dp[i] = max(dp[i], dp[i - 1], dp[compressed[start] - 1] + p)
    return dp[-1]

if __name__ == '__main__':
    print(main())
```

```cpp
int main() {
    int n = read();
    vector<tuple<int, int, int>> events;
    set<int> days;
    for (int i = 0; i < n; i++) {
        int a = read(), b = read(), p = read();
        events.push_back({a, -p, 0});
        events.push_back({b, p, a});
        days.insert(a);
        days.insert(b);
    }
    map<int, int> compressed;
    int i = 1;
    for (auto day : days) {
        compressed[day] = i++;
    }
    sort(events.begin(), events.end());
    vector<long long> dp(i + 1);
    for (auto [day, p, start] : events) {
        i = compressed[day];
        if (p < 0) {
            dp[i] = max(dp[i], dp[i - 1]);
        } else {
            dp[i] = max(dp[i], dp[i - 1]);
            dp[i] = max(dp[i], dp[compressed[start] - 1] + p);
        }
    }
    cout << dp[i] << endl;
}
```

## Removal Game

### Solution 1:  dynammic programming + interval

dp(i, j) = maximum score player can score compared to score of other player for the interval [i, j)

```cpp
int main() {
    int n = read();
    vector<long long> numbers(n);
    for (int i = 0; i < n; i++) {
        numbers[i] = readll();
    }
    vector<vector<long long>> dp(n + 1, vector<long long>(n + 1, LONG_LONG_MIN));
    for (int i = 0; i <= n; i++) {
        dp[i][i] = 0;
    }
    for (int len = 1; len <= n; len++) {
        for (int i = 0; i + len <= n; i++) {
            int j = i + len;
            dp[i][j] = max(dp[i][j], numbers[i] - dp[i + 1][j]);
            dp[i][j] = max(dp[i][j], numbers[j - 1] - dp[i][j - 1]);
        }
    }
    long long res = (dp[0][n] + accumulate(numbers.begin(), numbers.end(), 0LL)) / 2;
    cout << res << endl;
}
```

## Two Sets II

### Solution 1:  0/1 knapsack dp problem

dp[i][x] = count of ways for the subset of elements in 0...i with sum of x
dp[i][x] = dp[i-1][x] + dp[i-1][x-i]
Convert to 0/1 knapsack where you can either take the element or not take it.  It can be converted to this by realize that you just need to find the number of ways that the sum is equal to n*(n+1)/4, 

cause the summation of the natural number is n*(n+1)/2, but you just need a set to reach half the sum, then the other elements must be in other set and the sum of each set is equal.  So just need to look for half, can quickly check if it is odd, then there is 0 solutions. 

Then just iterate through all the possibilities with dynammic programming

```cpp
long long mod = int(1e9) + 7;

int main() {
    int n = read();
    int target = n * (n + 1) / 2;
    if (target & 1) {
        cout << 0 << endl;
        return 0;
    }
    target /= 2;
    vector<vector<long long>> dp(n + 1, vector<long long>(target + 1, 0));
    dp[0][0] = 1;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j <= target; j++) {
            dp[i][j] = dp[i - 1][j];
            if (j - i >= 0) dp[i][j] = (dp[i][j] + dp[i - 1][j - i]) % mod;
        }
    }
    cout << dp[n - 1][target] << endl;
}
```

## Elevator Rides

### Solution 1:  bitmask dp

dp[mask] = minimum pair of number of rides and then weight on last ride.  
So the best combination of these two values for taking a subset of weights is the best solution to subproblem, where subproblem is that of taking this subset of weights. 

```py
import math

def main():
    n, x = map(int, input().split())
    weights = list(map(int, input().split()))
    dp = [(math.inf, math.inf)] * (1 << n)
    dp[0] = (1, 0) # number rides, weight on last ride
    for mask in range(1, 1 << n):
        for i in range(n):
            if (mask >> i) & 1:
                prev_mask = mask ^ (1 << i)
                if dp[prev_mask][1] + weights[i] <= x:
                    dp[mask] = min(dp[mask], (dp[prev_mask][0], dp[prev_mask][1] + weights[i]))
                else:
                    dp[mask] = min(dp[mask], (dp[prev_mask][0] + 1, weights[i]))
    print(dp[-1][0])

if __name__ == '__main__':
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

##

### Solution 1: 

```py

```