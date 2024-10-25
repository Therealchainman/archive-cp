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

```cpp
vector<int> manacher(const string& s) {
    string t = "#";
    for (char ch : s) {
        t += ch;
        t += "#";
    }
    vector<int> parr = manacher_odd(t);
    return parr;
}
vector<int> manacher_odd(string& s) {
    int N = s.size();
    s = "$" + s + "^";
    vector<int> P(N + 2, 0);
    int l = 1, r = 1;
    for (int i = 1; i <= N; i++) {
        P[i] = max(0, min(r - i, P[l + (r - i)]));
        while (s[i - P[i]] == s[i + P[i]]) {
            P[i]++;
        }
        if (i + P[i] > r) {
            l = i - P[i];
            r = i + P[i];
        }
    }
    return vector<int>(P.begin() + 1, P.end() - 1);
}
```

## Manacher for static range queries

Example of how this manacher's algorithm works and what it returns
given string s = "abaaba"
You get the following string t = "#a#b#a#a#b#a#"
And you get parr = [1, 2, 1, 4, 1, 2, 7, 2, 1, 2, 1]

You can use manacher's array to perform range queries on the string to determine if the substring is palindromic.  
Let's call these static palindromic range queries

Important to note the range query is for range [l, r).  

```cpp
vector<int> marr;
vector<int> manacher(const string& s) {
    string t = "#";
    for (char ch : s) {
        t += ch;
        t += "#";
    }
    vector<int> parr = manacher_odd(t);
    return parr;
}
vector<int> manacher_odd(string& s) {
    int N = s.size();
    s = "$" + s + "^";
    vector<int> P(N + 2, 0);
    int l = 1, r = 1;
    for (int i = 1; i <= N; i++) {
        P[i] = max(0, min(r - i, P[l + (r - i)]));
        while (s[i - P[i]] == s[i + P[i]]) {
            P[i]++;
        }
        if (i + P[i] > r) {
            l = i - P[i];
            r = i + P[i];
        }
    }
    return vector<int>(P.begin() + 1, P.end() - 1);
}
// [l, r)
bool query(int l, int r) {
    return marr[l + r] > r - l;
}
```

call with where t is the string.

```cpp
marr = manacher(t);
```