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

const int inf = 1e15;

int32_t main() {
    int T = read();
    while (T--) {
        string left, right;
        cin >> left >> right;

        int n = right.length();
        left = string(n - left.length(), '0') + left;

        vector<vector<vector<vector<vector<int>>>>> dp(n + 1,
            vector<vector<vector<vector<int>>>>(2,
                vector<vector<vector<int>>>(2,
                    vector<vector<int>>(2,
                        vector<int>(2, -inf)
                    )
                )
            )
        );

        // (i, left_lower, left_upper, right_lower, right_upper)
        dp[0][1][1][1][1] = 0;

        for (int i = 0; i < n; i++) {
            int L = left[i] - '0';
            int R = right[i] - '0';

            for (int left_lower = 0; left_lower < 2; left_lower++) {
                for (int left_upper = 0; left_upper < 2; left_upper++) {
                    for (int d1 = 0; d1 < 10; d1++) {
                        if (left_lower && d1 < L) continue;
                        if (left_upper && d1 > R) break;

                        for (int right_lower = 0; right_lower < 2; right_lower++) {
                            for (int right_upper = 0; right_upper < 2; right_upper++) {
                                for (int d2 = 0; d2 < 10; d2++) {
                                    if (right_lower && d2 < L) continue;
                                    if (right_upper && d2 > R) break;

                                    int nleft_lower = left_lower && d1 == L;
                                    int nleft_upper = left_upper && d1 == R;
                                    int nright_lower = right_lower && d2 == L;
                                    int nright_upper = right_upper && d2 == R;

                                    dp[i + 1][nleft_lower][nleft_upper][nright_lower][nright_upper] =
                                        max(dp[i + 1][nleft_lower][nleft_upper][nright_lower][nright_upper],
                                            dp[i][left_lower][left_upper][right_lower][right_upper] + abs(d1 - d2)
                                        );
                                }
                            }
                        }
                    }
                }
            }
        }

        int res = 0;

        for (int left_lower = 0; left_lower < 2; left_lower++) {
            for (int left_upper = 0; left_upper < 2; left_upper++) {
                for (int right_lower = 0; right_lower < 2; right_lower++) {
                    for (int right_upper = 0; right_upper < 2; right_upper++) {
                        res = max(res, dp[n][left_lower][left_upper][right_lower][right_upper]);
                    }
                }
            }
        }

        cout << res << endl;
    }

    return 0L;
}
