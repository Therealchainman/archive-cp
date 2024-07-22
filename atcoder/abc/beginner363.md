# Atcoder Beginner Contest 363

## 

### Solution 1: 

```cpp

```

## F - Palindromic Expression 

### Solution 1:  recursion, memoization, palindrome

```cpp
int N;
map<int, string> memo;

int rev(int x) {
    int res = 0;
    while (x) {
        res = res * 10 + x % 10;
        x /= 10;
    }
    return res;
}

bool is_palindrome(int x) {
    return x == rev(x);
}

bool contains_zero(int x) {
    while (x > 0) {
        if (x % 10 == 0) return true;
        x /= 10;
    }
    return false;
}

string calc(int n) {
    if (memo.find(n) != memo.end()) return memo[n];
    if (!contains_zero(n) && is_palindrome(n)) {
        return memo[n] = to_string(n);
    }
    int x = 2;
    while (x * x <= n) {
        if (!contains_zero(x) && n % x == 0) {
            int y = rev(x);
            if ((n / x) % y== 0) {
                string res = calc(n / x / y);
                if (res != "-1") return memo[n] = to_string(x) + '*' + res + '*' + to_string(y);
            }
        }
        x++;
    }
    return memo[n] = "-1";
}

void solve() {
    cin >> N;
    string ans = calc(N);
    if (ans == "-1") {
        cout << -1 << endl;
    } else {
        cout << ans << endl;
    }
}

signed main() {
    solve();
    return 0;
}
```

## G - Dynamic Scheduling 

### Solution 1: 

```cpp

```