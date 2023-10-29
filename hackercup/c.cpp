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

string name = "meta_game_sample_input.txt";

int N;
vector<int> A;

void advance(int& p) {
    p = (p + 1) % (2 * N);
}

void decrement(int& p) {
    p = (p - 1 + 2 * N) % (2 * N);
}

bool check(int left, int right) {
    for (int i = 0; i < N; i++) {
        if (A[left] != A[right]) return false;
        advance(left);
        decrement(right);
    }
    return true;
}

int solve() {
    N = read();
    // cout << "N: " << N << endl;
    A.assign(2 * N, 0);
    for (int i = 0; i < N; i++) {
        A[i] = read();
    }
    for (int i = N; i < 2 * N; i++) {
        A[i] = read();
    }
    int window_N = N / 2;
    // A and B first windows
    int left_af = 0, right_af = N / 2 - 1;
    int left_bf = N, right_bf = N + N / 2 - 1;
    int count_f = 0;
    for (int i = 0; i < N / 2; i++) {
        if (A[i] < A[i + N]) count_f++;
        // cout << A[i] << " " << A[i + N] << endl;
    }
    // A and B second windows
    int left_as = N % 2 == 0 ? N / 2 : N / 2 + 1;
    int right_as = N - 1;
    int left_bs = N % 2 == 0 ? N + N / 2 : N + N / 2 + 1;
    int right_bs = 2 * N - 1;
    int count_s = 0;
    for (int i = N / 2 + (N % 2 == 0 ? 0 : 1); i < N; i++) {
        if (A[i] > A[i + N]) count_s++;
    }
    // cout << "count_f: " << count_f << endl;
    // cout << "count_s: " << count_s << endl;
    vector<int> index;
    if (count_s == window_N && count_f == window_N) {
        if (check(left_af, right_bs)) return 0;
    }
    for (int t = 1; t < 2 * N; t++) {
        // updates first window
        if (A[left_af] < A[left_bf]) count_f--;
        advance(left_af);
        advance(left_bf);
        // cout << "left_af: " << left_af << endl;
        // cout << "right_bf: " << right_bf << endl;
        // cout << "right_af: " << right_af << endl;
        advance(right_af);
        advance(right_bf);
        // cout << "right_bf: " << right_bf << endl;
        // cout << "right_af: " << right_af << endl;
        if (A[right_af] < A[right_bf]) count_f++;
        // cout << "count_f: " << count_f << endl;
        if (A[left_as] > A[left_bs]) count_s--;
        advance(left_as);
        advance(left_bs);
        advance(right_as);
        advance(right_bs);
        if (A[right_as] > A[right_bs]) count_s++;
        // cout << "count_s: " << count_s << endl;
        if (count_s == window_N && count_f == window_N) {
            if (check(left_af, right_bs)) return t;
        }
    }
    return -1;
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
        cout << "Case #" << i << ": " << solve() << endl;
    }
    return 0;
}

/*
problem solve

g++ "-Wl,--stack,1078749825" c.cpp -o main
*/
