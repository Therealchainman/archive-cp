# INTERSECTION OF LINES IN 3 DIMENSIONAL SPACE

This problem is a little more tricky than in 2 dimensions. 

There are three possibilities for the intersection of two lines in 3D space:
1. intersecting lines at a single point
2. parallel lines
3. skew lines


# succinct definition of skew lines
Skew lines are a pair of lines that are non-intersecting, non-parallel, and non-coplanar. This implies that skew lines can never intersect and are not parallel to each other. For lines to exist in two dimensions or in the same plane, they can either be intersecting or parallel.

## line intersection in python

```py
import math

def cross(u, v):
    """
    Returns the cross product of two 3D vectors u and v.
    """
    x = u[1] * v[2] - u[2] * v[1]
    y = u[2] * v[0] - u[0] * v[2]
    z = u[0] * v[1] - u[1] * v[0]
    return x, y, z

def norm(u):
    """
    Returns the norm of a 3D vector u.
    """
    return math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

def intersect(line1, line2):
    """
    Returns point if line segments line1 and line2 intersect in x < 0 region.
    returns None if the lines line1 and line2 do not intersect in x < 0 region or are parallel.
    """
    # Compute direction vectors and a point on each line
    v1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1], line1[1][2] - line1[0][2])
    # v1 = line1[1] - line1[0]
    p1 = line1[0]
    v2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1], line2[1][2] - line2[0][2])
    # v2 = line2[1] - line2[0]
    p2 = line2[0]

    # Compute normal vector to plane containing both lines
    n = cross(v1, v2)

    # Check if lines are parallel
    if norm(n) < 1e-6:
        return None

    # Compute intersection point
    p2p1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    p1p2 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    t1 = dot_product(cross(p2p1, v2), n) / dot_product(cross(v1, v2), n)
    t2 = dot_product(cross(p1p2, v1), n) / dot_product(cross(v2, v1), n)


    # point = p1 + t1 * v1
    # point2 = p2 + t2 * v2
    point1 = (p1[0] + t1*v1[0], p1[1] + t1*v1[1], p1[2] + t1*v1[2])
    point2 = (p2[0] + t2*v2[0], p2[1] + t2*v2[1], p2[2] + t2*v2[2])

    # check for skew lines
    if any(abs(v1 - v2) > 1e-6 for v1, v2 in zip(point1, point2)): return None

    # Check if intersection point is in x < 0 region
    if point1[0] < 0:
        return point1

    return None

def dot_product(u, v):
    """
    Returns the dot product of two 3D vectors u and v.
    """
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

def parallel(line1, line2):
    """
    Returns True if line1 and line2 are parallel.
    """
    # Compute direction vectors of lines
    v1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1], line1[1][2] - line1[0][2])
    v2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1], line2[1][2] - line2[0][2])
    # v1 = line1[1] - line1[0]
    # v2 = line2[1] - line2[0]

    # Compute cross product of direction vectors
    norm_vec = cross(v1, v2)

    # Check if cross product is zero
    if norm(norm_vec) < 1e-6:
        return True

    return False
```

## line intersection with numpy

```py

```

## line intersection with system of linear equations

```py

```
