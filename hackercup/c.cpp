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

string name = "sum_41_chapter_2_sample_input.txt";
// string name = "sum_41_chapter_2_validation_input.txt";
// string name = "sum_41_chapter_2_input.txt";

const int N = 1e9;
vector<int> spf;
void sieve(int n) {
    for (int i = 2; i <= n; i++) {
        if (spf[i] != 1) continue;
        for (int j = i; j <= n; j += i) {
            spf[j] = i;
        }
    }
}

vector<int> factorize(int x) {
    vector<int> factors;
    while (x > 1) {
        factors.push_back(spf[x]);
        x /= spf[x];
    }
    return factors;
}

void solve() {
    int P = read();
    vector<int> factors = factorize(P);
    int fsum = 0;
    for (int f : factors) fsum += f;
    if (fsum > 41) {
        cout << -1;
        return;
    }
    // use a min heap and just greedily merge the smallest two factors
    priority_queue<int, vector<int>, greater<int>> pq;
    for (int f : factors) pq.push(f);
    while (pq.size() > 1) {
        int a = pq.top(); pq.pop();
        int b = pq.top(); pq.pop();
        if (fsum - a - b + a * b > 41) {
            pq.push(a);
            pq.push(b);
            break;
        }
        pq.push(a * b);
    }
    factors.clear();
    while (!pq.empty()) {
        factors.push_back(pq.top());
        pq.pop();
    }
    while (fsum < 41) {
        factors.push_back(1);
        fsum++;
    }
    // then it will equal to 41 so print out all these factors
    cout << factors.size() << " ";
    for (int f : factors) cout << f << " ";
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T = read();
    spf.assign(N + 1, 1);
    sieve(N);
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
