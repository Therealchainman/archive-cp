# Suffix Sum

## Suffix Trick for Max Subarray Starting at Index

Given the left index of a subarray I want to find the maximum sum of any non-empty subarray starting from index i = l, ans = sufSum(l) - minSufSum(l + 1)

```cpp
bool check(const vector<int> &arr) {
    vector<int> ssum(N, 0), smin(N, 0);
    for (int i = N - 1; i >= 0; i--) {
        ssum[i] = arr[i];
        if (i + 1 < N) ssum[i] += ssum[i + 1];
        smin[i] = ssum[i];
        if (i + 1 < N) smin[i] = min(smin[i], smin[i + 1]);
    }
    int psum = 0;
    for (int i = 0; i + 2 < N; i++) {
        psum += arr[i];
        if (psum < 0) continue;
        if (ssum[i + 1] >= smin[i + 2]) return true;
    }
    return false;
}
```