# INNER AND OUTER PRODUCTS

## INNER PRODUCTS

When it comes to inner product I have thus far only dealt with vectors, and so the concept is very intuitive because one can easily visualize two vectors and how they get multiplied, and it is clear why the dot product of two vectors is defined the way it is. For v∗u
 it would basically be the length projection of vonto u
 (the part of v in direction of u) multiplied by the length of u
. So you basically have a measure of how much the vectors move in same direction.

Expects vectors of any dimension

```py
inner_product = lambda v1, v2: sum(x1*x2 for x1, x2 in zip(v1, v2))
```

```cpp
int64 dotProduct(int64 x1, int64 y1, int64 x2, int64 y2) {
    return x1 * x2 + y1 * y2;
}
```

### APPLICATION OF INNER PRODUCTS

An application of inner product is for suppose you have found that a point may be on a line segment, you can confirm if the point is on a line segment with the inner product. 
The way to determine a point may be on a line segment is if the outer product is equal to 0. This is also called colinear, you have found. 

outer product for two dimensions

```py
outer_product = lambda v1, v2: v1[0]*v2[1] - v1[1]*v2[0]
```
