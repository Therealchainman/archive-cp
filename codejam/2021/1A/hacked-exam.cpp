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
    int T, N, Q, m, num, den;
    string s;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> N >> Q;
        vector<string> vec;
        vector<int> M;
        string ans = "";
        for (int i = 0; i < N; i++)
        {
            cin >> s >> m;
            vec.push_back(s);
            M.push_back(m);
        }
        if (N == 1)
        {
            den = 1;
            int q = vec[0].size();
            num = max(q - M[0], M[0]);
            for (int i = 0; i < vec[0].size(); i++)
            {
                if (M[0] < q - M[0] && vec[0][i] == 'F')
                {
                    ans += 'T';
                }
                else if (M[0] < q - M[0])
                {
                    ans += 'F';
                }
                else
                {
                    ans += vec[0][i];
                }
            }
        }
        else if (N == 2)
        {
            den = 1;
            int m1 = M[0], m2 = M[1], l1 = 0, l2 = 0;
            string s1 = vec[0], s2 = vec[1];
            int n = s1.size();
            for (int i = 0; i < n; i++)
            {
                if (s1[i] == s2[i])
                {
                    l1++;
                }
                else
                {
                    l2++;
                }
            }
            int x = (m1 + m2 - l2) / 2;
            int y = m2 - x;
            num = max(x, l1 - x) + max(y, l2 - y);
            for (int i = 0; i < n; i++)
            {
                if (s1[i] == s2[i] && l1 - x > x)
                {
                    if (s1[i] == 'F')
                    {
                        ans += 'T';
                    }
                    else
                    {
                        ans += 'F';
                    }
                }
                else if (s1[i] == s2[i])
                {
                    ans += s1[i];
                }
                else if (l2 - y > y)
                {
                    ans += s1[i];
                }
                else
                {
                    ans += s2[i];
                }
            }
        }
        else
        {
            //hahahah
            num = 1;
            den = 1;
        }
        cout << "Case #" << t << ": " << ans << ' ' << num << '/' << den << endl;
    }
}
