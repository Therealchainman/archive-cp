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

int dfs(p2 p, vector<vector<int>> &M, set<p2> &vis, bool isVert)
{

    int R = M.size(), C = M[0].size();
    vis.insert(p);
    stack<pair<p2, bool>> stk;
    stk.push({p, true});
    p2 curp;
    int r, c;
    bool vert, hori;
    if (isVert)
    {
        int countDown = 0, counth = 0;
        while (!stk.empty())
        {
            tie(curp, countDown) = stk.top();
            tie(r, c) = curp;
            stk.pop();
            if (vert)
            {
                countDown++;
                if (r + 1 < R && M[r + 1][c] == 1)
                {
                    curp = {r + 1, c};
                    stk.push({curp, countDown + 1});
                }
                if (c - 1 >= 0 &&)
                {
                }
            }
        }
    }
    else
    {
    }
}

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
        int ans = 0;
        for (int r = 0; r < R; r++)
        {
            for (int c = 0; c < C; c++)
            {
                cin >> G;
                M[r][c] = G;
            }
        }
        set<p2> visited;
        // Find the L's with the longer vertical part
        for (int r = 0; r < R; r++)
        {
            for (int c = 0; c < C; c++)
            {
                if (M[r][c] == 1 && visited.count({r, c}) == 0)
                {
                    ans += dfs({r, c}, M, visited, true);
                }
            }
        }
        visited.clear();
        for (int r = 0; r < R; r++)
        {
            for (int c = 0; c < C; c++)
            {
                if (M[r][c] == 1 && visited.count({r, c}) == 0)
                {
                    ans += dfs({r, c}, M, visited, false);
                }
            }
        }
        cout << "Case #" << t << ": " << ans << endl;
    }
}