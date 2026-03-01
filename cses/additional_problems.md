# Additional Problems


## Shortest Subsequence

### Solution 1:

```cpp

```

## Distinct Values Sum

### Solution 1:

```cpp

```

## Distinct Values Splits

### Solution 1:

```cpp

```

## Swap Game

### Solution 1:

```cpp

```

## Beautiful Permutation II

### Solution 1:

```cpp

```

## Multiplication Table

### Solution 1:  math, binary search, greedy

```cpp
const int64 INF = numeric_limits<int64>::max();
int64 N;

bool possible(int64 target) {
    int64 cnt = 0;
    for (int r = 1; r <= N; ++r) {
        int64 c = min(target / r, N);
        cnt += c;
    }
    return cnt <= N * N / 2;
}

void solve() {
    cin >> N;
    int64 lo = 1, hi = INF;
    while (lo < hi) {
        int64 mid = lo + (hi - lo) / 2;
        if (possible(mid)) lo = mid + 1;
        else hi = mid;
    }
    cout << lo << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Bubble Sort Rounds I

### Solution 1:

```cpp

```

## Bubble Sort Rounds II

### Solution 1:

```cpp

```

## Nearest Campsites I

### Solution 1:

```cpp

```

## Nearest Campsites II

### Solution 1:

```cpp

```

## Advertisement

### Solution 1:

```cpp

```

## Special Substrings

### Solution 1:

```cpp

```

## Counting LCM Arrays

### Solution 1:

```cpp

```

## Square Subsets

### Solution 1:

```cpp

```

## Subarray Sum Constraints

### Solution 1:

```cpp

```

## Water Containers Moves

### Solution 1:

```cpp

```

## Water Containers Queries

### Solution 1:

```cpp

```

## Stack Weights

### Solution 1:

```cpp

```

## Maximum Average Subarrays

### Solution 1:

```cpp

```

## Subsets with Fixed Average

### Solution 1:

```cpp

```

## Two Array Average

### Solution 1:

```cpp

```

## Pyramid Array

### Solution 1: fenwick tree, coordinate compression, counting inversions

This is interesting but basically for each index i, you want to take the minimum of the number of inversions to the left and to the right of the current index.
That means you want to count how many elements that are greater to the left or to the right, cause these are those which you will have to swap with.

You can ignore smaller ones because those will already be counted.  They can all be done independently.

```cpp
int N;
vector<int> A;

template <typename T>
struct FenwickTree {
    vector<T> nodes;
    T neutral;

    FenwickTree() : neutral(T(0)) {}

    void init(int n, T neutral_val = T(0)) {
        neutral = neutral_val;
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, T val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    T query(int idx) {
        T result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }

    T query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : T(0);
    }
};

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    vector<int> sA(A.begin(), A.end());
    sort(sA.begin(), sA.end());
    FenwickTree<int> seg1, seg2;
    seg1.init(N); seg2.init(N);
    for (int i = 0; i < N; ++i) seg2.update(i + 1, 1);
    int64 ans = 0;
    for (int i = 0; i < N; ++i) {
        int idx = lower_bound(sA.begin(), sA.end(), A[i]) - sA.begin() + 1;
        int x = seg1.query(idx + 1, N);
        int y = seg2.query(idx + 1, N);
        ans += min(x, y);
        seg1.update(idx, 1);
        seg2.update(idx, -1);
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Permutation Subsequence

### Solution 1:

```cpp

```

## Bit Inversions

### Solution 1:

```cpp

```

## Writing Numbers

### Solution 1:

```cpp

```

## Letter Pair Move Game

### Solution 1:

```cpp

```

## Maximum Building I

### Solution 1:

```cpp

```

## Sorting Methods

### Solution 1:

```cpp

```

## Cyclic Array

### Solution 1:

```cpp

```

## List of Sums

### Solution 1:

```cpp

```
