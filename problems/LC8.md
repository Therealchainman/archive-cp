# 8. String to Integer (atoi)

### Solution: string parsing function

```c++
int myAtoi(string s) {
    long long res = 0;
    int start = find_if(s.begin(),s.end(),[](const auto& a) {
        return a=='+' || a=='-' || isdigit(a) || isalpha(a) || a=='.';
    })-s.begin();
    int sign = s[start]=='-' ? -1 : 1;
    start += (s[start]=='-' || s[start]=='+');
    for (int i = start;i<s.size();i++) {
        if (res>INT_MAX) break;
        if (isdigit(s[i])) {
            res = (res*10 + (s[i]-'0'));
        } else {break;}
    }
    res *= sign;
    if (res>0) {
        return min(res, (long long)INT_MAX);
    }
    return max(res, (long long)INT_MIN);
}
```