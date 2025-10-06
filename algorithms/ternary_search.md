# Ternary Search

## Description

Ternary search is a divide-and-conquer algorithm that can be used to find the maximum or minimum of a unimodal function. It works by dividing the search space into three parts and recursively searching in the part that contains the desired extremum.

## Algorithm

Works on an integer domain. l, r, m1, m2 are ints.

Seeks a maximum of a unimodal function on integers. Note the max in the tail scan and the if (f(m1) < f(m2)) update pattern which is for maximizing.

Finishes with a brute force loop over the last few integers.

```cpp
int64 ternarySearchMax(int l, int r) {
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

## Ternary Search for Minimum

Seeks a minimum of a unimodal function on integers. 

```cpp
// interesting example of unimodal function that is convex, with a global minimum
int64 f(int64 x) {
    int64 ans = max(0LL, ceil(h2, x) - a1) + max(0LL, a2 * x - h1 + 1);
    return ans;
}

int64 ternarySearchMin(int64 l, int64 r) {
    while (r - l > 3) {
        int64 m1 = l + (r - l) / 3;
        int64 m2 = r - (r - l) / 3;
        if (f(m1) <= f(m2)) r = m2 - 1; // minimum lies in [l, m2 - 1]
        else l = m1 + 1; // minimum lies in [m1 + 1, r]
    }
    int64 ans = INF;
    for (int64 i = l; i <= r; ++i) {
        ans = min(ans, f(i));
    }
    return ans;
}
```

## Continuous domain version

1. Works on a continuous domain. The parameter p is a real in [0, 1].
2. Seeks a minimum. The update rule keeps the side with the smaller value.
3. No tail brute force since the interval shrinks continuously. Fixed number of iterations controls precision.
4. num_iterations = 60 might work
5. Assumes the function is unimodal and convex (quadratic)

Just need to have some implementation for the function f. 

```cpp
const int ITERATIONS = 60;

long double ternarySearchMin(long double l, long double r) {
    for (int i = 0; i < ITERATIONS; ++i) {
        long double ml = (l * 2 + r) / 3;
        long double mr = (l + r * 2) / 3;
        if (f(ml) < f(mr)) r = mr;
        else l = ml;
    }
}
```

## Observations

Ternary search works if you can label a function as convex or concave, even in discrete space.  
The main definitions that give these is if the slope is always decreasing or always increasing.
If you can label a function as convex or concave, then you can use ternary search to find the maximum or minimum of that function.
If you  have a strictly decreasing sequence of integers, and you formulate a prefix sum from that, it will be convex.
If you have a strictly increasing sequence of integers, and you formulate a prefix sum from that, it will be concave.
The sum of convex functions is also convex. 