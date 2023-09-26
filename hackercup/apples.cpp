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
// string name = "two_apples_a_day_validation_input.txt";
string name = "two_apples_a_day_input.txt";

vector<int> arr;
int N;

bool check(int x, int k) {
    if (x <= 0) return false;
    int left = 0, right = 2 * N - 2;
    bool used = false;
    while (left < right) {
        if (arr[left] + arr[right] != k && !used) {
            if (arr[left] + x == k) {
                left++;
                used = true;
            } else if (x + arr[right] == k) {
                right--;
                used = true;
            } else {
                return false;
            }
        }
        else if (arr[left] + arr[right] != k) {
            return false;
        }
        left++;
        right--;
    }
    if (left == right && (arr[left] + x != k || used)) return false;
    return true;
}

int solve() {
    N = read();
    int M = 2 * N - 1;
    arr.resize(M);
    for (int i = 0; i < M; i++) {
        arr[i] = read();
    }
    int res = LLONG_MAX;
    if (N == 1) {
        return 1;
    }
    sort(arr.begin(), arr.end());
    int S = accumulate(arr.begin(), arr.end(), 0LL);
    vector<int> vals{arr[0] + arr.end()[-2], arr[0] + arr.end()[-1], arr[1] + arr.end()[-1]};
    for (int K : vals) {
        int x = K * N - S;
        if (check(x, K)) res = min(res, x);
    }
    if (res == LLONG_MAX) {
        res = -1;
    }
    return res;
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
