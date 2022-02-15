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

template <class T>
string toString(vector<T> &vec)
{
    stringstream res;
    copy(vec.begin(), vec.end(), ostream_iterator<T>(res, " "));
    return res.str().c_str();
}

// gcd to handles the zeros
ll gcd(ll a, ll b)
{
    if (b == 0 || a == 0)
    {
        return a + b;
    }
    return __gcd(a, b); // Using the inbuilt gcd
}

const int base = 256 * 1024;
ll tree[2 * base];

struct Test
{
    vector<vector<tuple<int, ll, ll>>> edges;
    vector<vector<pair<int, ll>>> queries;
    vector<ll> ans;

    void dfs(int node, int parent)
    {
        for (auto que : queries[node])
        {
            // query the segment tree
            ll g = tree[base + que.second];
            for (int x = base + que.second; x != 1; x >>= 1)
            {

                if (x % 2 == 1)
                {
                    g = gcd(g, tree[x - 1]);
                }
            }
            ans[que.first] = g;
        }
        int child;
        ll toll, limit;
        for (auto edge : edges[node])
        {
            child = get<0>(edge);
            limit = get<1>(edge);
            toll = get<2>(edge);
            if (child == parent)
            {
                continue;
            }
            // upate the segment tree
            tree[base + limit] = toll;
            for (int x = base + limit >> 1; x >= 1; x >>= 1)
            {
                tree[x] = gcd(tree[2 * x], tree[2 * x + 1]);
            }
            dfs(child, node);
            tree[base + limit] = 0;
            for (int x = base + limit >> 1; x >= 1; x >>= 1)
            {
                tree[x] = gcd(tree[2 * x], tree[2 * x + 1]);
            }
        }
    }

    void testCases(int t)
    {
        int N, Q, X, Y, C;
        ll A, L, W;
        cin >> N >> Q;
        edges.resize(N + 1);
        queries.resize(N + 1);
        ans.resize(Q);
        for (int i = 0; i < N - 1; i++)
        {
            cin >> X >> Y >> L >> A;
            edges[X].push_back({Y, L, A});
            edges[Y].push_back({X, L, A});
        }
        for (int i = 0; i < Q; i++)
        {
            cin >> C >> W;
            queries[C].emplace_back(i, W);
        }
        dfs(1, -1);
        cout << "Case #" << t << ": " << toString(ans) << endl;
    }
};
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        Test test;
        test.testCases(t);
    }
}