#include <bits/stdc++.h>
using namespace std;
#define int long long
#define x first
#define h second

inline int read() {
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

string name = "snakes_and_ladders_input.txt";

const int mod = 1e9 + 7;
vector<pair<int,int>> ladders;
vector<int> st;

bool comp(const int &a, const int &b) {
    return ladders[a].h > ladders[b].h;
}

void solve() {
    int N = read();
    ladders.assign(N, {0, 0});
    for (int i = 0; i < N; i++) {
        ladders[i].x = read();
        ladders[i].h = read();
    }
    int res = 0LL;
    sort(ladders.begin(), ladders.end());
    st.clear();
    for (int i = 0; i < N; i++) {
        while (!st.empty() && ladders[st.end()[-1]].h < ladders[i].h) {
            st.pop_back();
        }
        if (!st.empty()) {
            int j = lower_bound(st.begin(), st.end(), i, comp) - st.begin();
            for (int k = j; k < st.size(); k++) {
                int L = ladders[i].x - ladders[st[k]].x;
                int L_2 = (L * L) % mod;
                res = (res + L_2) % mod;
            }
        }
        st.push_back(i);
    }
    cout << res;
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
        cout << "Case #" << i << ": ";
        solve();
        cout << endl;
    }
    return 0;
}

/*
problem solve

g++ "-Wl,--stack,1078749825" c.cpp -o main
*/
