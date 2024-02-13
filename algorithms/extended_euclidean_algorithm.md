# Extended Euclidean Algorithm

## Solving Diophantine equaion with extended euclidean algorithm

Assume that this is not the case a = b = 0

```py
def extended_euclidean(a, b, x, y):
    if b == 0: return a, 1, 0
    g, x1, y1 = extended_euclidean(b, a % b, x, y)
    return g, y1, x1 - y1 * (a // b)
```

Example of how to implement it to solve the equation ax + by = c
if a or b or both are negative, you just need to include -a, -b into the function.  

```py
g, x, y = extended_euclidean(a, b, 0, 0)
if c % g != 0: # no solution
# find one solution
x *= 2 // g
y *= 2 // g
```