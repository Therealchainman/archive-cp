# Hashing

This is way to generate over a million unique hashes for indexing unique elements, so each bit represents a unique hash and element.

```py
mod1 = 1_000_000_007
mod2 = 998_244_353
vis = set()
bit = 1
while bit not in vis:
  if len(vis) == 1_000_000: 
    print('stopping')
    break
  vis.add(bit)
  bit = bit * mod1 % mod2
```

Finds a collision where subset of two elements has same hash sum as that of a single element.

```py
def f():
  vis2 = sorted(vis)
  for i, x in enumerate(vis):
      for j, y in enumerate(vis):
          if x + y in vis:
              print(i, j, x, y, x + y)
              return vis2.index(x + y)
  return -1
```

## Zobrist hashing

```cpp

```