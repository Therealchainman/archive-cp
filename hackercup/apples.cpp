#include <bits/stdc++.h>
using namespace std;
#define int long long

inline int read() {
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

// string name = "two_apples_a_day_sample_input.txt";
string name = "two_apples_a_day_validation_input.txt";
// string name = "two_apples_a_day_input.txt";

const int inf = pow(10, 17);

void solve(int t) {
    int N = read();
    int M = 2 * N - 1;
    vector<int> arr(M);
    for (int i = 0; i < M; i++) {
        arr[i] = read();
    }
    int res = inf;
    if (N == 1) {
        res = arr[0];
        cout << "Case #" << t << ": " << res << endl;
        return;
    }
    sort(arr.begin(), arr.end());
    map<int, int> counter;
    vector<int> sums(M);
    for (int i = 1; i < N; i++) {
        int s = arr[i] + arr[M - i];
        sums[i] = s;
        sums[M - i] = s;
        if (counter.find(s) == counter.end()) counter[s] = 0;
        counter[s]++;
    }
    if (counter.size() == 1) {
        int key = counter.begin()->first;
        if (key > arr[0]) res = min(res, key - arr[0]);
    }
    for (int i = 1; i < M; i++) {
        int x;
        if (i < N) {
            x = arr[M - i];
        } else {
            x = arr[M - i - 1];
        }
        int ps = arr[i] + x;
        int s = arr[i - 1] + x;
        counter[ps]--;
        if (counter.find(s) == counter.end()) counter[s] = 0;
        counter[s]++;
        if (counter[ps] == 0) counter.erase(ps);
        if (counter.size() == 1) {
            int key = counter.begin()->first;
            if (key > arr[i]) res = min(res, key - arr[i]);
        }
    }
    if (res == inf) {
        res = -1;
    }
    cout << "Case #" << t << ": " << res << endl;
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T = read();
    for (int i = 1; i <= T ; i++) {
        solve(i);
    }
    return 0;
}
