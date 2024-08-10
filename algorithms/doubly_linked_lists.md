# Doubly Linked Lists

I created this because doubly linked lists are very useful data structure in some scenarios.  Especially if you need to delete an element in an array at any index.  And maintain what are the next and previous elements.

## Initial

START = 0 always, and represent the start node, 0 because allows indexing at 0th.  And END = one greater than largest index in the data.  So it will signify the end of the doubly linked list.  So if nxt points to END, that means it is a terminal node.  And if prv points to START that means it is the first node in the linked list chain. 

```cpp
vector<int> nxt, prv;
int START, END;
nxt.assign(n + 2, n + 1);
prv.assign(n + 2, 0);
START = 0;
END = n + 1;
for (int i = 1; i <= n; i++) {
    nxt[i] = i + 1;
    prv[i] = i - 1;
}
```

## ERASE x from doubly linked list

```py
def erase(x):
    prv[nxt[x]] = prv[x]
    nxt[prv[x]] = nxt[x]
    prv[x] = START
    nxt[x] = END
```

```cpp
void erase(int x) {
    prv[nxt[x]] = prv[x];
    nxt[prv[x]] = nxt[x];
    prv[x] = START;
    nxt[x] = END;
}
```

## INSERT y after x in doubly linked list

```py
def insert(x, y): # insert y after x
    nxt[y] = nxt[x]
    prv[y] = x
    prv[nxt[x]] = y
    nxt[x] = y
```

```cpp
void insert(int x, int y) {
    nxt[y] = nxt[x];
    prv[y] = x;
    prv[nxt[x]] = y;
    nxt[x] = y;
}
```

## Example of a Doubly Linked List 

This uses the prv and nxt array to track the previous and next elements in the doubly linked list.  The doubly linked list is 1-indexed because it removes some annoying edge cases.  

One benefit of this script is how it initializes the doubly linked list and arrays all in the top, so it doesn't have to repeatedly do that. 

You instantly know if an index is out of bound if it is less than 1 or greater than N.  Cause 0 and N + 1 are out of bounds and represent that the neighbor element does not exist.

This is the important step where it deletes the element at index idx.

```cpp
prv[nxt[idx]] = prv[idx];
nxt[prv[idx]] = nxt[idx];
```

```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'

const int MAXN = 200'005;
int arr[MAXN], prv[MAXN], nxt[MAXN], used[MAXN];
int N;

bool ready(int i) {
    if (i < 1 || i > N) return false;
    return arr[prv[i]] + 1 == arr[i] || arr[nxt[i]] + 1== arr[i];
}

void solve() {
    cin >> N;
    for (int i = 1; i <= N; i++) {
        cin >> arr[i];
        prv[i] = i - 1;
        nxt[i] = i + 1;
        used[i] = 0;
    }
    arr[0] = arr[N + 1] = -2;
    priority_queue<pair<int, int>> pq;
    for (int i = 1; i <= N; i++) {
        if (ready(i)) {
            used[i] = 1;
            pq.emplace(arr[i], i);
        }
    }
    int val, idx;
    while (!pq.empty()) {
        tie(val, idx) = pq.top();
        pq.pop();
        prv[nxt[idx]] = prv[idx];
        nxt[prv[idx]] = nxt[idx];
        if (!used[prv[idx]] && ready(prv[idx])) {
            pq.emplace(arr[prv[idx]], prv[idx]);
            used[prv[idx]] = 1;
        }
        if (!used[nxt[idx]] && ready(nxt[idx])) {
            pq.emplace(arr[nxt[idx]], nxt[idx]);
            used[nxt[idx]] = 1;
        }
    }
    int unused = 0, mn = N;
    for (int i = 1; i <= N; i++) {
        unused += !used[i];
        mn = min(mn, arr[i]);
    }
    string ans = unused == 1 && mn == 0 ? "YES" : "NO";
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## In Python

```py
from heapq import heappop, heappush

MAXN = 2 * 10 ** 5 + 5
arr, prv, nxt, used = [0] * MAXN, [0] * MAXN, [0] * MAXN, [0] * MAXN

def main():
    N = int(input())
    def ready(i):
        if i < 1 or i > N: return False
        return arr[prv[i]] + 1 == arr[i] or arr[nxt[i]] + 1 == arr[i]
    lst = map(int, input().split())
    for i in range(1, N + 1):
        arr[i] = next(lst)
        prv[i] = i - 1
        nxt[i] = i + 1
        used[i] = 0
    arr[0] = arr[N + 1] = -2
    max_heap = []
    for i in range(1, N + 1):
        if ready(i):
            used[i] = 1
            heappush(max_heap, (-arr[i], i))
    while max_heap:
        v, i = heappop(max_heap)
        v = -v
        prv[nxt[i]] = prv[i]
        nxt[prv[i]] = nxt[i]
        if not used[prv[i]] and ready(prv[i]):
            used[prv[i]] = 1
            heappush(max_heap, (-arr[prv[i]], prv[i]))
        if not used[nxt[i]] and ready(nxt[i]):
            used[nxt[i]] = 1
            heappush(max_heap, (-arr[nxt[i]], nxt[i]))
    unused, val = 0, N
    for i in range(1, N + 1):
        if not used[i]:
            unused += 1
            val = arr[i]
    return unused == 1 and val == 0

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print("YES" if main() else "NO")
```