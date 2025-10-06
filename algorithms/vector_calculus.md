# Vector Calculus

## 2D space


```cpp
// p1 - p2
pair<long double, long double> sub(const pair<long double, long double> &p1, const pair<long double, long double> &p2) {
    return {p1.first - p2.first, p1.second - p2.second};
}

int64 distSquared(const pair<int64, int64> &p1, const pair<int64, int64> &p2) {
    int64 dx = p1.first - p2.first;
    int64 dy = p1.second - p2.second;
    return dx * dx + dy * dy;
}

// |p1 - p2|
long double dist(const pair<long double, long double> &p1, const pair<long double, long double> &p2) {
    pair<long double, long double> delta = sub(p1, p2);
    return sqrt(delta.first * delta.first + delta.second * delta.second);
}

// linear interpolation between p1 and p2, t in [0, 1]
pair<long double, long double> interpolate(const pair<long double, long double> &p1, const pair<long double, long double> &p2, long double t) {
    long double x = p1.first + (p2.first - p1.first) * t;
    long double y = p1.second + (p2.second - p1.second) * t;
    return {x, y};
}
```