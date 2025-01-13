# Practice

## 1400. Construct K Palindrome Strings

### Solution 1:  counting, parity, palindromes

```cpp
class Solution {
private:
    int encode(char ch) {
        return ch - 'a';
    }
public:
    bool canConstruct(string s, int k) {
        int N = s.size();
        vector<int> freq(26, 0);
        for (const char ch : s) {
            freq[encode(ch)]++;
        }
        int cnt = accumulate(freq.begin(), freq.end(), 0, [](int accum, int x) {
            return accum + (x & 1);
        });
        return k >= cnt && k <= N;
    }
};
```

## 2116. Check if a Parentheses String Can Be Valid

### Solution 1:  stack

```cpp
class Solution {
public:
    bool canBeValid(string s, string locked) {
        int N = s.size();
        if (N & 1) return false;
        stack<int> openBracket, unlocked;
        for (int i = 0; i < N; i++) {
            if (locked[i] == '0') {
                unlocked.push(i);
            } else if (s[i] == '(') {
                openBracket.push(i);
            } else {
                if (!openBracket.empty()) {
                    openBracket.pop();
                } else if (!unlocked.empty()) {
                    unlocked.pop();
                } else {
                    return false;
                }
            }
        }
        while (!openBracket.empty() && !unlocked.empty() && openBracket.top() < unlocked.top()) {
            openBracket.pop();
            unlocked.pop();
        }
        if (!openBracket.empty()) return false;
        return true;
    }
};
```

## 3223. Minimum Length of String After Operations

### Solution 1:  counting, parity, string

```cpp
class Solution {
private:
    int encode(char ch) {
        return ch - 'a';
    }
public:
    int minimumLength(string s) {
        vector<int> freq(26, 0);
        for (const char ch : s) {
            freq[encode(ch)]++;
        }
        int ans = 0;
        for (int x : freq) {
            if (!x) continue;
            ans++;
            if (x % 2 == 0) ans++;
        }
        return ans;
    }
};
```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```