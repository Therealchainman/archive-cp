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
#include <stdarg.h>
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

template <class T>
void printVec(vector<T> vec)
{
    cout << "[";
    for (auto val : vec)
    {
        cout << val << ",";
    }
    cout << "]" << endl;
}
template <class T>
void print2DVec(vector<vector<T>> &M)
{
    cout << "[" << endl;
    for (vector<T> row : M)
    {
        cout << "[";
        for (auto val : row)
        {
            cout << val << ",";
        }
        cout << "]," << endl;
    }
    cout << "]" << endl;
}
void printArray(int arr[], int n)
{
    cout << "[";
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << ",";
    }
    cout << "]" << endl;
}
template <class T>
void printMap(T mp)
{
    cout << "{";
    for (auto &v : mp)
    {
        cout << "{" << v.first << "," << v.second << "},";
    }
    cout << "}" << endl;
}

template <class T>
void printUSet(unordered_set<T> st)
{
    cout << "{";
    for (auto v : st)
    {
        cout << v << ",";
    }
    cout << "}" << endl;
}

template <class T>
void printNameElem(string name, T a)
{
    cout << name << ": " << a << endl;
}

template <class T>
void printElem(T a)
{
    cout << a << endl;
}

template <class T1, class T2, class T3>
void printVec2Pair(vector<pair<pair<T1, T2>, T3>> p)
{
    cout << "{";
    for (auto v : p)
    {
        cout << "((" << v.first.first << "," << v.first.second << ")," << v.second << "),";
    }
    cout << "}" << endl;
}

template <class T1, class T2>
void printVecPair(vector<pair<T1, T2>> p)
{
    cout << "{";
    for (auto v : p)
    {
        cout << "(" << v.first << "," << v.second << "),";
    }
    cout << "}" << endl;
}

typedef pair<int, int> p2;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T;
    ll A, B, C, shift, x, diff, ns;
    int s, h, m;
    ll M = 360LL * 12LL * 1e10;
    cin >> T;
    for (int t = 1; t <= T; t++)

    {
        cin >> A >> B >> C;
        vector<ll> ticks = {A, B, C};
        do
        {
            diff = ticks[1] - ticks[0];
            if (diff < 0)
            {
                diff += M;
            }
            printf("diff=%lld\n", diff);
            if (mod(diff, 11LL) == 0)
            {
                x = diff / 11;
                printf("x=%lld\n", x);
                shift = ticks[0] - x;
                if (shift < 0)
                {
                    shift += M;
                }
                bool isGood = true;
                for (int i = 0; i < 3; i++)
                {
                    if (mod(vector<int>{1, 12, 720}[i] * x + shift, M) != ticks[i])
                    {
                        isGood = false;
                    }
                }
                if (isGood)
                {
                    vector<int> ans;
                    ans.push_back(mod(x, 1e9));
                    x /= 1e9;
                    ans.push_back(mod(x, 60));
                    x /= 60;
                    ans.push_back(mod(x, 60));
                    x /= 60;
                    ans.push_back(x);
                    reverse(ans.begin(), ans.end());
                    if (ans[0] < 12)
                    {
                        h = ans[0], m = ans[1], s = ans[2], ns = ans[3];
                        printf("h=%d,m=%d,s=%d,ns=%lld\n", h, m, s, ns);
                    }
                }
            }

        } while (next_permutation(ticks.begin(), ticks.end()));

        cout << "Case #" << t << ": " << h << " " << m << " " << s << " " << ns << endl;
    }
}
