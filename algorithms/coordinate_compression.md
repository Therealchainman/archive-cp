# Coordinate Compression

Coordinate compression maps large or sparse values to a smaller range (e.g., from arbitrary integers to 0, 1, 2, ...) while preserving the relative order.

```cpp
sort(values.begin(), values.end());
values.erase(unique(values.begin(), values.end()), values.end());
for (int i = 0; i < N; ++i) {
    A[i] = lower_bound(values.begin(), values.end(), A[i]) - values.begin();
}
```