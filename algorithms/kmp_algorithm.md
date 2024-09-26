# KMP

Just know that for any pi[i] it tells you the longest suffix that ends at i (inclusive) that matches to the longest prefix.  and pi[0] = 0 always. 

## KMP Algorithm

Computes prefix array

```py
def kmp(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]: 
            j = pi[j - 1]
        if s[j] == s[i]: j += 1
        pi[i] = j
    return pi

parr = kmp(s)
```

```cpp
vector<int> kmp(const string& s) {
    int N = s.size();
    vector<int> pi(N, 0);
    for (int i = 1; i < N; i++) {
        int j = pi[i - 1];
        while (j > 0 && s[i] != s[j]) {
            j = pi[j - 1];
        }
        if (s[j] == s[i]) j++;
        pi[i] = j;
    }
    return pi;
}
```

## KMP algortihm (Knuth-Morris-Pratt)

This is example of using kmp algorithm to find the number of occurrences of pattern in a text.  

This one is old what is this doing, how is lcp in here working? 

```py
text = input()
pat = input()
n, m = len(text), len(pat)
def lcp(s):
    dp = [0] * m
    j = 0
    for i in range(1, m):
        if s[i] == s[j]:
            j += 1
            dp[i] = j
            continue
        while j > 0 and s[i] != s[j]:
            j -= 1
        dp[i] = j
        if s[i] == s[j]:
            j += 1
    return dp
def kmp(text, pat):
    j = cnt = 0
    for i in range(n):
        while j > 0 and text[i] != pat[j]:
            j = lcp_arr[j - 1]
        if text[i] == pat[j]:
            j += 1
        if j == m:
            cnt += 1
            j = lcp_arr[j - 1]
    return cnt
lcp_arr = lcp(pat)
res = kmp(text, pat)
print(res)
```

