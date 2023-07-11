# binary search

## binary search to solve quadratic function

This is example of how was able to binary search for the k values in a quadratic function using binary search algorithm.  

Finding if there is a solution to this equation 1 + x + x^2 = n, where x is the unknown variable.  So just need to find at what x it is equal to n.  But x > 0,  so you are only searching one side of quadratic equation that is why binary search works.  

The area that is actually be searched is monotonic and that is why binary search will work. 

![images](images/quadratic.png)

```py
quadratic = lambda k: 1 + k + k**2
left, right = 2, 1_000_000_000
while left < right:
    mid = (left + right) >> 1
    if quadratic(mid) == n: return print("Yes")
    if quadratic(mid) <= n: left = mid + 1
    else: right = mid
```