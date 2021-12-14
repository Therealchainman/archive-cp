
# Part 1 

Brute force solution with time of O(nm), where n = len(positions) and m = mx-mn

```py
positions = list(map(int, f.read().split(',')))
mx, mn = max(positions), min(positions)
lo, hi, initial = mn, mx, 1458460
minFuel = reduce(lambda minFuel, pos: min(minFuel, sum(abs(pos - crabPos) for crabPos in positions)), range(lo, hi + 1), initial)
print(minFuel)
```

binary search solution with O(nlog(m)) time

```py
positions = list(map(int, f.read().split(',')))
lo, hi = min(positions), max(positions)
def fuelCost(target):
    return sum(abs(target-pos) for pos in positions)
while lo < hi:
    mid = (lo + hi + 1) >> 1
    if fuelCost(mid) < fuelCost(mid-1):
        lo = mid
    else:
        hi = mid - 1
print(fuelCost(lo))
```
Solution using median, this is done in O(n) time
```py
positions = list(map(int, f.read().split(',')))
medianPos = int(median(positions))
print(sum(abs(medianPos - pos) for pos in positions))
```
Using numpy for solution 
```py
positions = list(map(int, f.read().split(',')))
minFuel = int(abs(positions - np.median(positions)).sum())
print(minFuel)
```
# Part 2


```py
positions = list(map(int, f.read().split(',')))
mx, mn = max(positions), min(positions)
lo, hi, initial = mn, mx, 1161136310
minFuel = reduce(lambda minFuel, pos: min(minFuel, sum(abs(pos-crabPos)*(abs(pos-crabPos)+1)//2 for crabPos in positions)), range(lo, hi + 1), initial)
print(minFuel)
```


```py
positions = list(map(int, f.read().split(',')))
hi, lo = max(positions), min(positions)
def fuelCost(target):
    return sum(abs(target-pos)*(abs(target-pos)+1)//2 for pos in positions)
while lo < hi:
    mid = (lo + hi + 1) >> 1
    if fuelCost(mid) < fuelCost(mid-1):
        lo = mid
    else:
        hi = mid - 1
print(fuelCost(lo))
```
Using mean to solve the problem, in O(n) time using statistics
```py
positions = list(map(int, f.read().split(',')))
meanPos = int(mean(positions))
minFuel = sum(abs(pos-meanPos)*(abs(pos-meanPos)+1)//2 for pos in positions)
print(minFuel)
```

Using mean in numpy 

```py
positions = np.array(list(map(int, f.read().split(','))))
minFuel = int(sum(n*(n+1)/2 for n in abs(positions - int(np.mean(positions)))))
print(minFuel)
```