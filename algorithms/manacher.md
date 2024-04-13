# Manacher's Algorithm

Gives you the longest palindrome centered at each index of a string in O(n) time. 
You can determine if there is a palindrome of a length by floor division by 2, and if parr[m] < len_ then there is no palindrome of that length.

```py
def manacher(s):
    t = "#".join(s)
    t = "#" + t + "#"
    res = manacher_odd(t)
    return res
 
def manacher_odd(s):
    n = len(s)
    s = "$" + s + "^"
    p = [0] * (n + 2)
    l, r = 1, 1
    for i in range(1, n + 1):
        p[i] = max(0, min(r - i, p[l + (r - i)]))
        while s[i - p[i]] == s[i + p[i]]:
            p[i] += 1
        if i + p[i] > r:
            l, r = i - p[i], i + p[i]
    return p[1:-1]

parr = manacher(s)
for i in range(len(parr)):
    if i % 2 == 0:
        parr[i] -= 1
    parr[i] //= 2
```