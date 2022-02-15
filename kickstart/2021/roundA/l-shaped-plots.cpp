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

/*
This is basically brute force solution, 
where just looking through each possible L shape, by counting down, 
up, left, right.  That is consider the long segments as I find them from scanning downwards,
scanning upwards, scanning both sideways.  Each time a long segment is greater than or equal to 4. 
I will start a search to the two sideways directions if there is a value of 1, there
And consider the limit to be the length of long segment divided by 2 because that is the longest the
short side can be.   Once you do that you count the length of short segment up to the limit which is the long segment divided by 2. 
And yeah that should work.
I need to describe better later.  
*/

int searchRight(int r, int c, vector<vector<int>> &M, int limit)
{
    int R = M.size(), C = M[0].size();
    int sSegment = 0;
    while (sSegment < limit && c < C && M[r][c] == 1)
    {
        sSegment++;
        c++;
    }
    // printf("limit=%d\n", limit);
    // printf("sSegment=%d\n", sSegment);
    return sSegment - 1;
}

int searchLeft(int r, int c, vector<vector<int>> &M, int limit)
{
    int R = M.size(), C = M[0].size();
    int sSegment = 0;
    while (sSegment < limit && c >= 0 && M[r][c] == 1)
    {
        sSegment++;
        c--;
    }
    // printf("limit=%d\n", limit);
    // printf("sSegment=%d\n", sSegment);
    return sSegment - 1;
}

int searchUp(int r, int c, vector<vector<int>> &M, int limit)
{
    int R = M.size(), C = M[0].size();
    int sSegment = 0;
    while (sSegment < limit && r >= 0 && M[r][c] == 1)
    {
        sSegment++;
        r--;
    }
    // printf("limit=%d\n", limit);
    // printf("sSegment=%d\n", sSegment);
    return sSegment - 1;
}

int searchDown(int r, int c, vector<vector<int>> &M, int limit)
{
    int R = M.size(), C = M[0].size();
    int sSegment = 0;
    while (sSegment < limit && r < R && M[r][c] == 1)
    {
        sSegment++;
        r++;
    }
    // printf("limit=%d\n", limit);
    // printf("sSegment=%d\n", sSegment);
    return sSegment - 1;
}

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, R, C, G, cnt;
    cin >> T;
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
        // countDown
        for (int c = 0; c < C; c++)
        {
            cnt = 0;
            for (int r = 0; r < R; r++)
            {
                if (M[r][c] == 1)
                {
                    cnt++;
                }
                else
                {
                    cnt = 0;
                }
                if (cnt >= 4)
                {
                    if (c - 1 >= 0 && M[r][c - 1] == 1)
                    {
                        ans += searchLeft(r, c, M, cnt / 2);
                    }
                    if (c + 1 < C && M[r][c + 1] == 1)
                    {
                        ans += searchRight(r, c, M, cnt / 2);
                    }
                }
            }
        }
        // countUp
        for (int c = 0; c < C; c++)
        {
            cnt = 0;
            for (int r = R - 1; r >= 0; r--)
            {
                if (M[r][c] == 1)
                {
                    cnt++;
                }
                else
                {
                    cnt = 0;
                }
                if (cnt >= 4)
                {

                    if (c - 1 >= 0 && M[r][c - 1] == 1)
                    {
                        ans += searchLeft(r, c, M, cnt / 2);
                    }
                    if (c + 1 < C && M[r][c + 1] == 1)
                    {

                        ans += searchRight(r, c, M, cnt / 2);
                    }
                }
            }
        }
        // countRight
        for (int r = 0; r < R; r++)
        {
            cnt = 0;
            for (int c = 0; c < C; c++)
            {
                if (M[r][c] == 1)
                {
                    cnt++;
                }
                else
                {
                    cnt = 0;
                }
                if (cnt >= 4)
                {
                    if (r - 1 >= 0 && M[r - 1][c] == 1)
                    {
                        ans += searchUp(r, c, M, cnt / 2);
                    }
                    if (r + 1 < R && M[r + 1][c] == 1)
                    {
                        ans += searchDown(r, c, M, cnt / 2);
                    }
                }
            }
        }
        // countLeft
        for (int r = 0; r < R; r++)
        {
            cnt = 0;
            for (int c = C - 1; c >= 0; c--)
            {
                if (M[r][c] == 1)
                {
                    cnt++;
                }
                else
                {
                    cnt = 0;
                }
                if (cnt >= 4)
                {
                    if (r - 1 >= 0 && M[r - 1][c] == 1)
                    {
                        ans += searchUp(r, c, M, cnt / 2);
                    }
                    if (r + 1 < R && M[r + 1][c] == 1)
                    {
                        ans += searchDown(r, c, M, cnt / 2);
                    }
                }
            }
        }

        cout << "Case #" << t << ": " << ans << endl;
    }
}