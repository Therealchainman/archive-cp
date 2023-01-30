#include <bits/stdc++.h>
using namespace std;

int memo[1301];
int thres = 10000;

int grundy(int coins) {
	if (memo[coins] != -1) return memo[coins];
	if (coins <= 2) return 0;
	vector<int> grundy_numbers(thres, 0);
	for (int i = 1; i <= coins/2; i++) {
		if (i == coins - i) continue;
		grundy_numbers[grundy(i) ^ grundy(coins - i)] = 1;
	}
	for (int grundy_number = 0; grundy_number < thres; grundy_number++) {
		if (grundy_numbers[grundy_number] == 0) {
			return memo[coins] = grundy_number;
		}
	}
	return -1;
}

int main() {
	cin.tie(0)->sync_with_stdio(0);
	int n, t;
	cin >> t;
	memset(memo, -1, sizeof(memo));
	grundy(1300);
	while (t--) {
		cin >> n;
		if (n <= 1300) {
			cout << (memo[n] > 0 ? "first" : "second") << endl;
		} else {
			cout << "first" << endl;
		}
	}
	return 0;
}
