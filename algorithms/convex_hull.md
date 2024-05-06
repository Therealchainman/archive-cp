# convex hull

## Convex Hull Algorithm


```cpp
// can be int sometimes
struct P {
    double x, y;

    bool operator<(const P &b) const {
        return make_pair(x, y) < make_pair(b.x, b.y);
    }
};
vector<P> points;
for (int rep = 0; rep < 2; rep++) {
    vector<int> hull; // hull that holds the indices of the points
    for (int i = 0; i < points.size(); i++) {
        auto [x3, y3] = points[i];
        while (hull.size() >= 2) {
            auto [x1, y1] = points[hull.end()[-2]];
            auto [x2, y2] = points[hull.end()[-1]];
            if ((x3 - x1) * (y2 - y1) <= (x2 - x1) * (y3 - y1)) break;
            hull.pop_back();
        }
        hull.push_back(i);
    }
    // hull.pop_back();
    reverse(points.begin(), points.end());
}
```

Where I originally found this implementation of convex hull was four coordinates that were doubles. And this line was like this

```cpp
            if ((x3 - x1) * (y2 - y1) < (x2 - x1) * (y3 - y1)) break;

```