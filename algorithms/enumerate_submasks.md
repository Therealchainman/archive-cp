# Enumerating over all submasks of a bitmask

## Implementation

time complexity is O(3^n)

m is the bitmask for which you want to enumerate over all it's submasks


```py
s = m
while s > 0:
    s = (s - 1) & m
```

```cpp
for (int submask = mask; submask > 0; submask = (submask - 1) & mask) {
    // do something with submask
}
```

## Set difference for submasks

Given integer s and t which represent a bitmask that represents a set of elements.  If t is a submask of s, then you can find the integer x = s - t, which represents the set difference.  That is all elements in s that were not in t. 