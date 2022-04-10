"""
First step is let's review how matrix exponentiation works

this is my thoughts on it. 

I need some ingredients
"""

#include <bits/stdc++.h>
using namespace std;
 
const int N = 15, md = 1e9 + 7;
 
int sz = 7;
 
int mt[N][N] = {
    {1, 2, 2, 2, 1, 1, 1},
    {0, 0, 0, 1, 0, 0, 0},
    {0, 1, 0, 1, 0, 1, 0},
    {0, 0, 1, 1, 0, 0, 1},
    {0, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 0, 1},
    {0, 2, 2, 2, 1, 1, 1}
};
 
int result[N][N], c[N][N], d[N][N];
 
void mul(int a[N][N], int b[N][N]) {
    for (int i = 0; i < sz; ++i)
        for (int j = 0; j < sz; ++j) {
            c[i][j] = a[i][j];
            d[i][j] = b[i][j];
            a[i][j] = 0;
        }
    for (int i = 0; i < sz; ++i)
        for (int j = 0; j < sz; ++j)
            for (int k = 0; k < sz; ++k)
                a[i][j] = (a[i][j] + 1LL * c[i][k] * d[k][j]) % md;
}
 
int solve(long long n) {
    if (n == 1) {
        return 9;
    }
    if (n == 2) {
        return 13;
    }
    n -= 3;
    for (int i = 0; i < sz; ++i)
        result[i][i] = 1;
    while (n > 0) {
        if (n & 1)
            mul(result, mt);
        mul(mt, mt);
        n /= 2;
    }
    int rs = 1;
    rs = (rs + 1LL * result[0][0] * 13) % md;
    rs = (rs + 1LL * result[0][4] * 6) % md;
    rs = (rs + 1LL * result[0][5] * 3) % md;
    rs = (rs + 1LL * result[0][6] * 2) % md;
    rs = (rs + 1LL * result[0][7] * 9) % md;
    rs = (rs + 1LL * result[0][8] * 4) % md;
    rs = (rs + 1LL * result[0][9] * 1) % md;
    return rs;
}
 
int main() {
    long long n;
    cin >> n;
    cout << solve(n) << endl;
    return 0;
}