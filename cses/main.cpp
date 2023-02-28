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
    int n = read(), q = read();
    vector<int> arr(n);
    for (int i = 0; i < n; i++) {
        arr[i] = read();
    }
    int max_power_two = 18;
    vector<int> lg(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        lg[i] = lg[i / 2] + 1;
    }
    vector<vector<int>> sparse_table(n, vector<int>(n + 1, INT_MAX));
    for (int j = 0; j <= max_power_two; j++) {
        for (int i = 0; i + (1 << j) <= n; i++) {
            if (j == 0) {
                sparse_table[i][j] = arr[i];
            }
            else {
                sparse_table[i][j] = min(sparse_table[i][j - 1], sparse_table[i + (1 << (j - 1))][j - 1]);
            }
        }
    }
    auto query = [&](int left, int right) -> int {
        int length = right - left + 1;
        int power_two = lg[length];
        return min(sparse_table[left][power_two], sparse_table[right - (1 << power_two) + 1][power_two]);
    };
    for (int i = 0; i < q; i++) {
        int a = read(), b = read();
        cout << query(a - 1, b - 1) << endl;
    }
    return 0;
}