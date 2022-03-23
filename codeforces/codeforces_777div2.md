# Codeforces Round 777 Div2

This feels like a greedy contest, just need to find simple tricks to make the questions astronomically simpler.  
such as using the 2 by 2 square for madoka and the elegant gift.  Or like using the 1 by 2 and 2 by 1 rectangles
to paint for childish pranks

## Madoka and Math Dad

```py
def find_num(start):
	num, i, digit_sum = 0, start, 0
	values = [1,2]
	while digit_sum < n:
		num = (num*10) + values[i%2]
		digit_sum += values[i%2]
		i+=1
	if digit_sum==n:
		return num
	return -1
 
def solve():
	first = find_num(1)
	if first != -1:
		return first
	return find_num(0)
 
 
if __name__ == '__main__':
	T = int(input())
	for _ in range(T):
	    n = int(input())
	    print(solve())
```

## Madoka and the Elegant Gift

```py
def solve():
    for r in range(1, n):
        for c in range(1, m):
            if grid[r][c]+grid[r-1][c]+grid[r][c-1]+grid[r-1][c-1]==3: return False
    return True


if __name__ == '__main__':
    T = int(input())
    ANS = []
    for _ in range(T):
        n, m = map(int,input().split())
        grid = [list(map(int, list(input()))) for _ in range(n)]
        ANS.append("YES" if solve() else "NO")
    print("\n".join(ANS))
```

## Madoka and Childish Pranks

```py
def solve():
    if grid[0][0] == '1': return "-1"
    instructions = []
    for r in range(n-1,-1,-1):
        for c in range(m-1,-1,-1): 
            if grid[r][c] == '1':
                if c > 0:
                    instructions.append(" ".join(map(str,(r+1, c, r+1, c+1))))
                else:
                    instructions.append(" ".join(map(str, (r, c+1, r+1, c+1))))
    return instructions
 
 
if __name__ == '__main__':
    T = int(input())
    ANS = []
    for _ in range(T):
        n, m = map(int,input().split())
        grid = [input() for _ in range(n)]
        ANS.append(solve())
    for ans in ANS:
        if ans == "-1":
            print(-1)
        elif not ans:
            print(0)
        else:
            print(len(ans))
            print("\n".join(ans))
```