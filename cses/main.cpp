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

int main() {
    int n = read();
    vector<tuple<int, int, int>> events;
    set<int> days;
    for (int i = 0; i < n; i++) {
        int a = read(), b = read(), p = read();
        events.push_back({a, -p, 0});
        events.push_back({b, p, a});
        days.insert(a);
        days.insert(b);
    }
    map<int, int> compressed;
    int i = 1;
    for (auto day : days) {
        compressed[day] = i++;
    }
    sort(events.begin(), events.end());
    vector<long long> dp(i + 1);
    for (auto [day, p, start] : events) {
        i = compressed[day];
        if (p < 0) {
            dp[i] = max(dp[i], dp[i - 1]);
        } else {
            dp[i] = max(dp[i], dp[i - 1]);
            dp[i] = max(dp[i], dp[compressed[start] - 1] + p);
        }
    }
    cout << dp[i] << endl;
}
