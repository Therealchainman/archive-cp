#include <bits/stdc++.h>


using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

#define Point pair<int, int>
#define x first
#define y second

long long square(long long x) {
    return x * x;
}

long long euclidean_dist(Point &p1, Point &p2) {
    return square(p1.x - p2.x) + square(p1.y - p2.y);
}

long long divide(vector<Point> &points, vector<Point> &points_y) {
    int n = points.size();
    if (n <= 1) {
        return LLONG_MAX;
    }
    vector<Point> left_points(points.begin(), points.begin() + n / 2);
    vector<Point> right_points(points.begin() + n / 2, points.end());
    vector<Point> left_points_y;
    vector<Point> right_points_y;
    int mid_x = left_points.back().x;
    int mid_y = left_points.back().y;
    for (auto &p : points) {
        if (make_pair(p.x, p.y) <= make_pair(mid_x, mid_y)) {
            left_points_y.push_back(p);
        } else {
            right_points_y.push_back(p);
        }
    }
    long long d = min(divide(left_points, left_points_y), divide(right_points, right_points_y));
    vector<Point> strip;
    for (auto &p : points_y) {
        if (abs(p.x - mid_x) < d) {
            strip.push_back(p);
        }
    }
    for (int i = 0; i < strip.size(); ++i) {
        for (int j = i + 1; j < strip.size() && strip[j].y - strip[i].y < d; ++j) {
            d = min(d, euclidean_dist(strip[i], strip[j]));
        }
    }
    return d;
}

int main() {
    int n = read();
    vector<Point> points(n);
    for (int i = 0; i < n; i++) {
        int x = read(), y = read();
        points[i] = {x, y};
    }
    sort(points.begin(), points.end(), [](auto &a, auto &b) {return make_pair(a.x, a.y) < make_pair(b.x, b.y);});
    vector<Point> points_y = points;
    sort(points_y.begin(), points_y.end(), [](auto &a, auto &b) {return a.y < b.y;});
    cout << divide(points, points_y) << endl;
    return 0;
}