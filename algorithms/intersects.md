# intersections

## intersection of line segments on number line

Can use the max(x1, x2) <= min(y1, y2)
for segments defined by endpoints x, y, where x <= y

```py
intersects = lambda a, b: max(segments[a][0], segments[b][0]) <= min(segments[a][1], segments[b][1])
```