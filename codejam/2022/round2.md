# Google Codejam Round 2

## Spiraling Into Control

### Solution 1: Find math pattern in circle, label each ring in spiral starting from central at 0, and compute the number of moves need to save and use shortcuts when necessary

```py

```

## Pixelated Circle

### Solution 1:  Solve test case 1 with set theory, brute force to compute the circles in sets, then find the outer join of the sets, basically find all elements that are not in both sets. 

```py
from math import sqrt
from itertools import product
def main():
    row = int(input())
    fill_circle = set()
    fill_circle_wrong = set()
    def draw_circle_filled_wrong(R):
        for r in range(R+1):
            draw_circle_perimeter(r)
    def draw_circle_perimeter(R):
        for x in range(-R, R+1):
            y = round(sqrt(R*R-x*x))
            fill_circle_wrong.add((x,y))
            fill_circle_wrong.add((x,-y))
            fill_circle_wrong.add((y,x))
            fill_circle_wrong.add((-y,x))
    def draw_circle_filled(R):
        for x, y in product(range(-R,R+1), repeat=2):
            if round(sqrt(x*x+y*y)) <= R:
                fill_circle.add((x,y))
    draw_circle_filled(row)
    draw_circle_filled_wrong(row)
    return len(fill_circle^fill_circle_wrong)
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Spiraling Into Control

### Solution 1:

```py

```

## Spiraling Into Control

### Solution 1:

```py

```