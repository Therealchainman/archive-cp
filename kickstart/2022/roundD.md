# Google Kickstart 2022 Round D

## Summary

##  P1: Image Labeler

### Solution 1:  greedy + math + sort

```py
def main():
    n, m = map(int,input().split())
    arr = sorted(list(map(float,input().split())))
    arr1 = arr[:n-m+1]
    arr2 = arr[n-m+1:]
    n1 = len(arr1)
    median = arr1[n1//2] if n1%2 else (arr1[n1//2]+arr1[n1//2-1])/2
    return sum(arr2) + median
    
    
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
```

## P2: Maximum Gain

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```