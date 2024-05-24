# Monotonic Stack

## Finding the next greater element

This is implementation that finds the nearest greater element, you can do it from either side, this finds the nearest to the left, So given 5, 4, 3, 4
If you are at the last 4, the nearest greater element is the 5. 

```cpp
stack<int> stk;
for (int i = 0; i < N; i++) {
    while (!stk.empty() && A[i] >= A[stk.top()]) {
        stk.pop();
    }
    L[i] = i - (stk.empty() ? -1 : stk.top());
    stk.push(i);
}
```

A slight variant that finds the nearest greater than or equal element

```cpp
stack<int> stk;
for (int i = N - 1; i >= 0; i--) {
    while (!stk.empty() && A[i] > A[stk.top()]) {
        stk.pop();
    }
    R[i] = (stk.empty() ? N : stk.top()) - i;
    stk.push(i);
}
```