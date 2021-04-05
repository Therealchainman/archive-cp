#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <vector>
#include <regex>
#include <set>
#include <chrono>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <ctime>
#include <cassert>
#include <complex>
#include <string>
#include <cstring>
#include <chrono>
#include <random>
#include <array>
#include <bitset>
#define rep(i, n) for (i = 0; i < n; ++i)
#define REP(i, k, n) for (i = k; i <= n; ++i)
#define REPR(i, k, n) for (i = k; i >= n; --i)
#define pb push_back
#define all(a) a.begin(), a.end()
#define fastio               \
    ios::sync_with_stdio(0); \
    cin.tie(0);              \
    cout.tie(0)
#define ll long long
#define uint unsigned long long
#define inf 0x3f3f3f3f3f3f3f3f
#define mxl INT64_MAX
#define mnl INT64_MIN
#define mx INT32_MAX
#define mn INT32_MIN
#define endl '\n'
using namespace std;
using namespace std::chrono;

ll mod(ll a, ll b)
{
    return (a % b + b) % b;
}

typedef pair<int, int> p2;

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, R, C, G;
    cin >> T;
    vector<p2> dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (int t = 1; t <= T; t++)
    {
        cin >> R >> C;
        vector<vector<int>> M(R, vector<int>(C, 0));
        for (int r = 0; r < R; r++)
        {
            for (int c = 0; c < C; c++)
            {
                cin >> G;
                M[r][c] = G;
            }
        }
        int ans = 0;
        for (int r = 0; r < R; r++)
        {
            for (int c = 0; c < C; c++)
            {
                for (p2 p : dirs)
                {
                    int dr, dc, nr, nc;
                    tie(dr, dc) = p;
                    nr = r + dr;
                    nc = c + dc;
                    if (nc >= 0 && nc < C && nr >= 0 && nr < R && abs(M[nr][nc] - M[r][c]) > 1)
                    {
                        if (M[nr][nc] > M[r][c])
                        {
                            ans += M[nr][nc] - M[r][c] - 1;
                            M[r][c] = M[nr][nc] - 1;
                        }
                        else
                        {
                            ans += M[r][c] - M[nr][nc] - 1;
                            M[nr][nc] = M[r][c] - 1;
                        }
                    }
                }
            }
        }
        cout << "Case #" << t << ": " << ans << endl;
    }
}