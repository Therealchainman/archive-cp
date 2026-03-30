# Sorting and Searching

## Distinct Numbers

### Solution 1:

```cpp
```

## Apartments

### Solution 1:

```cpp
```

## Ferris Wheel

### Solution 1:

```cpp
```

## Concert Tickets

### Solution 1:

```cpp
```

## Restaurant Customers

### Solution 1:

```cpp
```

## Movie Festival

### Solution 1:

```cpp
```

## Sum of Two Values

### Solution 1:

```cpp
```

## Maximum Subarray Sum

### Solution 1:

```cpp
```

## Stick Lengths

### Solution 1:

```cpp
```

## Missing Coin Sum

### Solution 1:

```cpp
```

## Collecting Numbers

### Solution 1:

```cpp
```

## Collecting Numbers II

### Solution 1:

```cpp
```

## Playlist

### Solution 1:

```cpp
```

## Towers

### Solution 1:

```cpp
```

## Traffic Lights

### Solution 1:

```cpp
```

## Distinct Values Subarrays

### Solution 1:

```cpp
```

## Distinct Values Subsequences

### Solution 1:

```cpp
```

## Josephus Problem I

### Solution 1:

```cpp
```

## Josephus Problem II

### Solution 1:

```cpp
```

## Nested Ranges Check

### Solution 1:

```cpp
```

## Nested Ranges Count

### Solution 1:

```cpp
```

## Room Allocation

### Solution 1:

```cpp
```

## Factory Machines

### Solution 1:

```cpp
```

## Tasks and Deadlines

### Solution 1:

```cpp
```

## Reading Books

### Solution 1:

```cpp
```

## Sum of Three Values

### Solution 1:

```cpp
```

## Sum of Four Values

### Solution 1: pair sums, map, two sum reduction

Reduce `4sum` into `2sum on pair sums`.

For each pair `(i, j)` with `j > i`, we want some earlier pair `(k, l)` such that:

`A[k] + A[l] + A[i] + A[j] = X`

So while fixing `i`, check whether `X - A[i] - A[j]` already exists in a map of pair sums.

The important invariant is that the map only stores pairs ending before `i`. We first query all pairs `(i, j)` with `j > i`, and only after that we insert all pairs `(j, i)` with `j < i`. That guarantees the four indices are distinct.

Time complexity is `O(n^2 log n)` with `map`, and memory is `O(n^2)` in the worst case.

```cpp
int N, X;
vector<int> A;

void solve() {
    cin >> N >> X;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    map<int, pair<int, int>> mp;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            int target = X - A[i] - A[j];
            if (mp.find(target) != mp.end()) {
                auto [k, l] = mp[target];
                cout << k + 1 << " " << l + 1 << " " << i + 1 << " " << j + 1 << endl;
                return;
            }
        }
        for (int j = 0; j < i; ++j) {
            mp[A[i] + A[j]] = {j, i};
        }
    }
    cout << "IMPOSSIBLE" << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Nearest Smaller Values

### Solution 1:

```cpp
```

## Subarray Sums I

### Solution 1:

```cpp
```

## Subarray Sums II

### Solution 1:

```cpp
```

## Subarray Divisibility

### Solution 1:

```cpp
```

## Distinct Values Subarrays II

### Solution 1:

```cpp
```

## Array Division

### Solution 1:

```cpp
```

## Movie Festival II

### Solution 1:

```cpp
```

## Maximum Subarray Sum II

### Solution 1:

```cpp
```
