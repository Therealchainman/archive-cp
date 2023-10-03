# KMP

## KMP algortihm (Knuth-Morris-Pratt)

This is example of using kmp algorithm to find the number of occurrences of pattern in a text.  

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