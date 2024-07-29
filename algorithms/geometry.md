# geometry

## outer product or cross product

```py
outer_product = lambda v1, v2: v1[0]*v2[1] - v1[1]*v2[0]
```

# function to test if a point is on the line segment p1p2

```py
def is_boundary(p, p1, p2):
    # is p on the boundary of p1p2
    x, y = p
    x1, y1 = p1
    x2, y2 = p2
    v1 = (x2 - x1, y2 - y1)
    v2 = (x - x1, y - y1)
    return outer_product(v1, v2) == 0 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
```

## function to test if line segment L1 (p1, p2) intersects L2 (p3, p4)

There is important condition is that this doesn't work if the lines intersect at one of the end points of the line segment.  this is a specialized function for line segment intersection that can work for point in polygon with ray casting.  Because you can guarantee that the lines will never intersect at a end point of the line segment.

```py
def intersects(p1, p2, p3, p4):
    for _ in range(2):
        v1, v2, v3 = (p2[0]-p1[0], p2[1]-p1[1]), (p3[0]-p1[0], p3[1]-p1[1]), (p4[0]-p1[0], p4[1]-p1[1])
        outer_prod1 = outer_product(v1, v2)
        outer_prod2 = outer_product(v1, v3)
        if (outer_prod1 < 0 and outer_prod2 < 0) or (outer_prod1 > 0 and outer_prod2 > 0): return False
        p1, p2, p3, p4 = p3, p4, p1, p2
    return True
```

## Determine if circles intersect

returns true if circle with center (x1, y1) and radius r1 intersects circle with center (x2, y2) with radius r2.

```cpp
bool intersection(int x1, int y1, int x2, int y2, int r1, int r2) {
    double d = sqrt((x1 - x2) * (x1 - x2)+ (y1 - y2) * (y1 - y2));
    if (d <= r1 - r2 || d <= r2 - r1 || d < r1 + r2 || d == r1 + r2) return true;
    return false;
}
```