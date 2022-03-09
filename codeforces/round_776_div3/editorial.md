# Codeforces Round 776 Div3

## A. Deletions of Two Adjacent Letters

I actually struggled with this on during the virtual contest because I missed the case when
len(S)==1 and S==c

```py
def func(S,c):
	for i in range(0,len(S),2):
		if S[i]==c: return True
	return False
T = int(input())
for _ in range(T):
	S = input()
	c = input()
	if func(S,c):
		print("YES")
	else:
		print("NO")
```


## B. DIV + MOD

This is just a very mathematical problem. I wrote down the math on paper and broke it down into intervals of length a

```py
def func(x,a):
	return x//a + x%a
T = int(input())
for _ in range(T):
	left, right, a = map(int, input().split())
	best = func(right,a)
	if right//a - left//a > 0:
		best = max(best, right//a-1+a-1)
	print(best)
```

## D. Twist the Permutation

I was a little bit uncertain, but it seems that it is always possible to use cycle shifts to get the current array, so never need to return -1.
I came up with an algorithm that is basically O(n^2) but is fast enough because n is 1000.  The idea was working backwards and reducing to the current
array that is important, finding the number of cycle shifts to get the last element into the correct position. 

```py
def solve(n, arr):
    res = [0]*n
    target_arr = list(range(n))
    for i in range(n-1,-1,-1):
        loc = arr.index(i)
        target_loc = target_arr.index(i)
        num_cycles = (loc-target_loc)%len(arr)
        if loc==target_loc:
            target_arr.remove(i)
            arr.remove(i)
            continue
        res[i] = num_cycles
        target_arr = target_arr[len(arr)-num_cycles:] + target_arr[:-num_cycles]
        target_arr.remove(i)
        arr.remove(i)
    return res
 
T = int(input())
ANS = []
for _ in range(T):
    n = int(input())
    arr = list(map(lambda x: int(x)-1,input().split()))
    res = solve(n,arr)
 
    ANS.append(" ".join(map(str,res)))
print('\n'.join(ANS))
```

Optimized solution I'm learning

```py
import sys
input = sys.stdin.readline
 
t=int(input())
for tests in range(t):
    n=int(input())
    A=list(map(int,input().split()))
 
    ANS=[]
 
    for i in range(n):
        #print(A)
        x=A.index(n-i)
        if x+1==len(A):
            ANS.append(0)
        else:
            ANS.append(x+1)
        A=A[x+1:]+A[:x]
 
    print(*ANS[::-1])
```