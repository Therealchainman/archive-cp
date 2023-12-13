# Enumerating over all submasks of a bitmask

## Implementation

time complexity is O(3^n)

```py
s = m
while s > 0:
    s = (s - 1) & m
```