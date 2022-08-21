## Manacher's algorithm

This gives TLE on 4 testcases in cses, so while it should work, it is too slow for cses

```py
def main():
    neutral = '#'
    s = '$' + input()
    arr = []
    for ch in s:
        arr.extend([ch, neutral])
    arr.append('^')
    n = len(arr)
    p = [0]*n
    left = right = 1
    max_length = start = end = 0
    for i in range(1,n-1):
        p[i] = max(0, min(right-i, p[left + (right - i)]))
        while arr[i-p[i]] == arr[i+p[i]]:
            p[i] += 1
        if i+p[i] > right:
            left, right = i-p[i], i+p[i]
        if p[i] > max_length:
            start, end = left+1, right
            max_length = p[i]
    return ''.join(filter(lambda x: x!=neutral, arr[start:end]))
if __name__ == '__main__':
    print(main())
```

This solutions passes very quickly online

```cpp
#include <bits/stdc++.h>
using namespace std;
const char neutral = '#';

int main() {
    string s;
    // freopen("input.txt","r",stdin);
    cin>>s;
    int n = s.size();
    // freopen("output.txt", "w", stdout);
    s = '$' + s;
    vector<char> arr(2*n+3);
    for (int i = 0;i<=n;i++) {
        arr[2*i] = s[i];
        arr[2*i+1] = neutral;
    }
    arr.end()[-1] = '^';
    vector<int> p(2*n+1);
    int left = 1, right = 1, max_length = 0, start = 0, end = 0;
    for (int i = 1;i<=2*n;i++) {
        p[i] = max(0, min(right-i, p[left+(right-i)]));
        while (arr[i-p[i]] == arr[i+p[i]]) {
            p[i]++;
        }
        if (i+p[i] > right) {
            left = i-p[i], right = i+p[i];
        }
        if (p[i] > max_length) {
            start = left+1, end = right;
            max_length = p[i];
        }
    }
    string longest_palindrome = "";
    for (int i = start;i<end;i++) {
        if (arr[i]==neutral) continue;
        longest_palindrome += arr[i];
    }
    cout<<longest_palindrome<<endl;
}
```