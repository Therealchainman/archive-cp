#include <bits/stdc++.h>
using namespace std;


class Solution {
private:
    const int MAXN = 4005;
    static const long long MOD = 1e9 + 7;
    static bool precomputed;
    static vector<vector<long long>> dp;
    long long inv(int i, long long m) {
    return i <= 1 ? i : m - (long long)(m/i) * inv(m % i) % m;
    }
    vector<long long> fact, inv_fact;
    void factorials(int n, long long m) {
        fact.assign(n + 1, 1);
        inv_fact.assign(n + 1, 0);
        for (int i = 2; i <= n; i++) {
            fact[i] = (fact[i - 1] * i) % m;
        }
        inv_fact.end()[-1] = inv(fact.end()[-1], m);
        for (int i = n - 1; i >= 0; i--) {
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % m;
        }
    }
    long long choose(int n, int r, long long m) {
        if (n < r) return 0;
        return (fact[n] * inv_fact[r] % m) * inv_fact[n - r] % m;
    }
    void calc(int n, long long m) {
        if (precomputed) return;
        factorials(n, m);
        dp.assign(n + 1, vector<long long>(n + 1, 0));
        dp[0][0] = 1;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                dp[i][j] = choose(i, j, m);
                if (j > 0) dp[i][j] = (dp[i][j] + dp[i][j - 1]) % m;
            }
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= i; j++) {
                for (int k = 0; k <= j; k++) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - k] * choose(j, k, m)) % m;
                }
            }
        }
        precomputed = true;
    }
public:
    int lengthAfterTransformations(string s, int t) {
        calc(MAXN, MOD);
        for (int i = 0; i <= 6; i++) {
            cout << dp[6][i] << " ";
        }
        return 0;
    }
};

bool Solution::precomputed = false;