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
void printSet(T st)
{
    cout << "{";
    for (auto v : st)
    {
        cout << *v << ",";
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

int T, X, Y;

int recurse(int i, string &mural, vector<vector<int>> &memo, char prev)
{
    if (i >= mural.size())
    {
        return 0;
    }
    if (prev != '?' && memo[i][(prev == 'C' ? 0 : 1)] != INT16_MAX)
    {
        return memo[i][(prev == 'C' ? 0 : 1)];
    }
    if (mural[i] == '?')
    {
        if (prev == 'C')
        {
            return memo[i][0] = min(recurse(i + 1, mural, memo, 'C'), recurse(i + 1, mural, memo, 'J') + X);
        }
        if (prev == 'J')
        {
            return memo[i][1] = min(recurse(i + 1, mural, memo, 'C') + Y, recurse(i + 1, mural, memo, 'J'));
        }
        return memo[i][0] = memo[i][1] = min(recurse(i + 1, mural, memo, 'C'), recurse(i + 1, mural, memo, 'J'));
    }
    if (i != 0)
    {
        if (prev == 'C' && mural[i] == 'J')
        {
            return memo[i][0] = X + recurse(i + 1, mural, memo, mural[i]);
        }
        else if (prev == 'J' && mural[i] == 'C')
        {
            return memo[i][1] = Y + recurse(i + 1, mural, memo, mural[i]);
        }
        return memo[i][(prev == 'C' ? 0 : 1)] = recurse(i + 1, mural, memo, mural[i]);
    }
    return memo[i][0] = memo[i][1] = recurse(i + 1, mural, memo, mural[i]);
}

typedef pair<int, int> p2;

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    string mural;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> X >> Y >> mural;
        vector<vector<int>> memo(mural.size(), vector<int>(2, INT16_MAX));
        recurse(0, mural, memo, '?');
        int cost = min(memo[0][0], memo[0][1]);
        cout << "Case #" << t << ": " << cost << endl;
    }
}