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

#include <bits/stdc++.h>
using namespace std;
 
typedef long long i64;
 
#ifdef Local
#include "debug.h"
#else
#define debug(...) 0
#endif
 
constexpr int K = 500;
 
void upd(i64 &x, i64 y) {
    x = max(x, y);
}
 
void solve() {
    int n;
    cin >> n;
    vector<vector<int>> g(n + 1);
    for (int i = 0, u, v; i < n - 1; i++) {
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    vector<int> sz(n + 1, 1);
    auto dfs1 = [&](auto self, int u, int f) -> void {
        for (int i = 0; i < g[u].size(); i++) {
            int v = g[u][i];
            if (v == f) continue;
            self(self, v, u);
            sz[u] += sz[v];
        }
    };
    dfs1(dfs1, 1, 0);
 
    using vii = vector<vector<i64>>;
 
    auto dfs2 = [&](auto self, int u, int f) -> vii {
 
        vii a(2, vector<i64>(2));
        a[0][1] = 1, a[1][1] = 0;
 
        i64 sum = 1;
        for (int v : g[u]) {
            if (v == f) continue;
 
            auto b = self(self, v, u);
            int n = a[0].size() - 1;
            int m = b[0].size() - 1;
 
            vii c(2, vector<i64>(min(n + m + 1, K + 1)));
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= m; j++) {
                    upd(c[0][i], a[0][i] + b[1][j] + 2 * sz[v] * sum);
                    upd(c[1][i], a[1][i] + b[0][j] + 2 * sz[v] * sum);
                    if (i + j <= K) {
                        upd(c[0][i + j], a[0][i] + b[0][j] + 2 * sz[v] * sum - i * j);
                        upd(c[1][i + j], a[1][i] + b[1][j] + 2 * sz[v] * sum - 2 * i * j);
                    }
                }
            }
            sum += sz[v];
 
            a = c;
        }
        return a;
    };
    auto dp = dfs2(dfs2, 1, 0);
 
    i64 ans = 0;
    for (int z = 0; z < 2; z++) {
        for (i64 i : dp[z]) {
            upd(ans, i);
        }
    }
    cout << ans << '\n';
}
 
int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    int t;
    cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}