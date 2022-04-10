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

const string IMPOSSIBLE = "IMPOSSIBLE";

bool isPossible(string &S, string &E)
{
    int cntS = 0, cntE = 0;
    for (int i = 1; i < E.size(); i++)
    {
        if (E[i - 1] == '0' && E[i] == '1')
        {
            cntE++;
        }
    }
    if (cntE == 0)
    {
        return true;
    }
    for (int i = 1; i < S.size(); i++)
    {
        if (S[i - 1] == '0' && S[i] == '1')
        {
            cntS++;
        }
    }
    return cntS == cntE;
}

template <class T>
void output(int t, T out)
{
    cout << "Case #" << t << ": " << out << endl;
}

string notBit(string &s)
{
    string res = "";
    bool prev = false;
    for (int i = 0; i < s.size(); i++)
    {
        if (s[i] == '0')
        {
            prev = true;
            res += '1';
        }
        else if (prev)
        {
            res += '0';
        }
    }
    if (res.size() == 0)
    {
        res += '0';
    }
    return res;
}

string doubleVal(string &s)
{
    string res = "";
    if (s.size() == 1 && s[0] == '0')
    {
        res += '0';
        return res;
    }
    res += s + '0';
    return res;
}

const int LIMIT = 1e9;
typedef pair<int, int> p2;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, oper;
    string S, E, B;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> S >> E;
        queue<pair<string, int>> q;
        unordered_set<string> seen;
        q.emplace(S, 0);
        seen.insert(S);
        int cnt = 0;
        bool isPossible = false;
        while (!q.empty())
        {
            tie(B, oper) = q.front();
            cnt++;
            if (cnt == LIMIT)
            {
                break;
            }
            if (B == E)
            {
                output(t, oper);
                isPossible = true;
                break;
            }
            q.pop();
            if (B.size() > 32)
            {
                continue;
            }
            string n = notBit(B);
            if (seen.count(n) == 0)
            {
                q.emplace(n, oper + 1);
                seen.insert(n);
            }
            string d = doubleVal(B);
            if (seen.count(d) == 0)
            {

                q.emplace(d, oper + 1);
                seen.insert(d);
            }
        }
        if (!isPossible)
        {
            output(t, IMPOSSIBLE);
        }
    }
}
