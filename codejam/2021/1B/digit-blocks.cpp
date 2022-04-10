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

int find9(vector<int> &cnt, int in, int B)
{
    int maxx = -1;
    int maxxi = -1;
    for (int i = 0; i < cnt.size(); i++)
    {
        if (cnt[i] > maxx && cnt[i] < B)
        {
            maxx = cnt[i];
            maxxi = i;
        }
    }
    return maxxi;
}

int find8(vector<int> &cnt, int in, int B)
{
    int maxx = -1;
    int maxxi = -1;
    for (int i = 0; i < cnt.size(); i++)
    {
        if (cnt[i] > maxx && cnt[i] < B - 1)
        {
            maxx = cnt[i];
            maxxi = i;
        }
    }
    if (maxxi == -1)
    {
        for (int i = 0; i < cnt.size(); i++)
        {
            if (cnt[i] > maxx && cnt[i] < B)
            {
                maxx = cnt[i];
                maxxi = i;
            }
        }
    }
    return maxxi;
}

int find(vector<int> &cnt, int in, int B)
{
    int minn = B + 1;
    int minni = -1;
    for (int i = 0; i < cnt.size(); i++)
    {
        if (cnt[i] < minn)
        {
            minn = cnt[i];
            minni = i;
        }
    }
    return minni;
}

typedef pair<int, int> p2;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, D, B, N, cand;
    ll in;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> N >> B;
        vector<int> cnt(N, 0);
        for (int i = 0; i < N * B; i++)
        {
            cin >> in;
            if (in == 9)
            {
                cand = find9(cnt, in, B);
                cnt[cand]++;
                cout << cand << endl;
                continue;
            }
            if (in == 8)
            {
                cand = find8(cnt, in, B);
                cnt[cand]++;
                cout << cand << endl;
                continue;
            }
            cand = find(cnt, in, B);
            cnt[cand]++;
            cout << cand << endl;
        }
    }
}
