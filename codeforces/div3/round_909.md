# Codeforces Round 909 div 3

## E. Queue Sort

### Solution 1: greedy, sorted after min element, index of min element

```cpp
int solve() {
    int n;
    cin >> n;
    vector<int> arr(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    int x = *min_element(arr.begin(), arr.end());
    int i = 0;
    while (arr[i] > x) {
        i++;
    }
    int res = i++;
    for (i; i < n; i++) {
        if (arr[i] < arr[i - 1]) return -1;
    }
    return res;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        cout << solve() << endl;
    }
    return 0;
}
```

## F. Alex's whims

### Solution 1: trick, observation, tree, undirected graph

Just move the last node node n to node d, and then it will have a distance equal to d from node 1. 

```py
def main():
    n, q = map(int, input().split())
    for i in range(2, n + 1):
        print(i - 1, i)
    u = n - 1
    for _ in range(q):
        d = int(input())
        if d == u:
            print(-1, -1, -1)
        else:
            print(n, u, d)
        u = d

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## 

### Solution 1: greedy

```cpp

```