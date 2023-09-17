# INTERACTIVE PROBLEMS

Example for problem, flush may not be necessary, it worked without, any thing above doesn't matter, can keep normal template for fast i/o

Surprisingly there is not anything extra needed for interactive problems in codeforces. 

## Python Example

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    psum = [0]*(n + 1)
    for i in range(n):
        psum[i+1] = psum[i] + arr[i]
    left, right = 0, n - 1
    while left < right:
        mid = (left + right) >> 1
        size = mid - left + 1
        print('?', size, *range(left + 1, mid + 2), flush = True)
        x = int(input())
        if x > psum[mid + 1] - psum[left]:
            right = mid
        else:
            left = mid + 1
    print('!', left + 1, flush = True)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C++ Example

Sometimes C++ is needed to pass time limit, but it is not always necessary.

```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long

inline int read() {
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    int t = read();
    while (t--) {
        int n = read();
        int mex = n;
        for (int i = 0; i < n; i++) {
            int x = read();
            if (mex < i) continue;
            if (x != i) {
                mex = i;
            }
        }
        while (true) {
            cout << mex << endl;
            int resp = read();
            if (resp == -1) break;
            if (resp == -2) break;
            mex = resp;
        }
    }
    return 0;
}
```