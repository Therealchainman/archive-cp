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

const long long is_query = -(1LL<<62);
struct line {
    long long m, b;
    mutable function<const line*()> succ;
    bool operator<(const line& rhs) const {
        if (rhs.b != is_query) return m < rhs.m;
        const line* s = succ();
        if (!s) return 0;
        long long x = rhs.m;
        return b - s->b < (s->m - m) * x;
    }
};

struct dynamic_hull : public multiset<line> { // will maintain upper hull for maximum
    const long long inf = LLONG_MAX;
    bool bad(iterator y) {
        auto z = next(y);
        if (y == begin()) {
            if (z == end()) return 0;
            return y->m == z->m && y->b <= z->b;
        }
        auto x = prev(y);
        if (z == end()) return y->m == x->m && y->b <= x->b;

		/* compare two lines by slope, make sure denominator is not 0 */
        long long v1 = (x->b - y->b);
        if (y->m == x->m) v1 = x->b > y->b ? inf : -inf;
        else v1 /= (y->m - x->m);
        long long v2 = (y->b - z->b);
        if (z->m == y->m) v2 = y->b > z->b ? inf : -inf;
        else v2 /= (z->m - y->m);
        return v1 >= v2;
    }
    void insert_line(long long m, long long b) {
        auto y = insert({ m, b });
        y->succ = [=] { return next(y) == end() ? 0 : &*next(y); };
        if (bad(y)) { erase(y); return; }
        while (next(y) != end() && bad(next(y))) erase(next(y));
        while (y != begin() && bad(prev(y))) erase(prev(y));
    }
    long long eval(long long x) {
        auto l = *lower_bound((line) { x, is_query });
        return l.m * x + l.b;
    }
};

struct Point {
    long long x, y, a;
};

int main(){
	int n = read();
	vector<long long> dp(n + 1);
    vector<Point> points(n);
	for (int i = 0; i < n; ++i) {
        long long x = readll(), y = readll(), a = readll();
        points[i] = {x, y, a};
	}
    sort(points.begin(), points.end(), [](auto &a, auto &b) {return a.y > b.y;});
	dynamic_hull cht;
	cht.insert_line(0, 0);
	for (int i = 1; i < n + 1; i++) {
        long long x = points[i - 1].x, y = points[i - 1].y, a = points[i - 1].a;
		// eval for x value
		dp[i] = x*y - a + cht.eval(y);
		// insert (m, b) line
		// for minimize take - (m, b), negative of slope and y-intercept
		cht.insert_line(-x, dp[i]);
	}
	auto mx = max_element(dp.begin(), dp.end());
	cout << *mx << endl;
}