# Ternary Search

## Description

Ternary search is a divide-and-conquer algorithm that can be used to find the maximum or minimum of a unimodal function. It works by dividing the search space into three parts and recursively searching in the part that contains the desired extremum.

## Algorithm

```cpp
int64 ternarySearch(int l, int r) {
    while (r - l > 3) {
        int m1 = l + (r - l) / 3;
        int m2 = r - (r - l) / 3;
        if (f(m1) < f(m2)) l = m1 + 1;
        else r = m2 - 1;
    }
    int64 ans = 0;
    for (int i = l; i <= r; i++) {
        ans = max(ans, f(i));
    }
    return ans;
}
```

## Observations

Ternary search works if you can label a function as convex or concave, even in discrete space.  
The main definitions that give these is if the slope is always decreasing or always increasing.
If you can label a function as convex or concave, then you can use ternary search to find the maximum or minimum of that function.
If you  have a strictly decreasing sequence of integers, and you formulate a prefix sum from that, it will be convex.
If you have a strictly increasing sequence of integers, and you formulate a prefix sum from that, it will be concave.
The sum of convex functions is also convex. 