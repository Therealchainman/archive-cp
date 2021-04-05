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

struct p3
{
    int i, j, k;

    bool operator==(const p3 &p) const
    {
        return this->i == p.i && this->j == p.j && this->k == p.k;
    }
};

struct hashFunc
{
    template <class T>
    size_t operator()(const T &p) const
    {
        auto hash1 = hash<T>()(p.i);
        auto hash2 = hash<T>()(p.j);
        auto hash3 = hash<T>()(p.k);
        return hash1 ^ hash2 ^ hash3;
    }
};

string makeString(vector<int> A)
{
    stringstream ss;
    for (int i = 0; i < A.size(); i++)
    {
        if (i != 0)
        {
            ss << " ";
        }
        ss << A[i];
    }
    return ss.str();
}

void ask(int i, int j, int k)
{
    cout << i << " " << j << " " << k << " " << endl;
}

int main()
{
    // ios_base::sync_with_stdio(false);
    // cin.tie(0);
    // cout.tie(0);
    int T, N, Q, i, j, k, L, med;
    cin >> T >> N >> Q;
    if (T == -1 || N == -1 || Q == -1)
    {
        exit(1);
    }
    for (int t = 1; t <= T; t++)
    {
        queue<int> q;
        vector<int> res;
        unordered_set<int> seen;
        res.push_back(1);
        res.push_back(2);
        for (int i = 3; i <= N; i++)
        {
            int lo = 0, hi = res.size() - 1;
            while (lo < hi)
            {
                int mid1 = lo + (hi - lo) / 3;
                int mid2 = mid1 + (hi - lo) / 3;
                if (mid2 == mid1)
                {
                    mid2++;
                }
                ask(res[mid1], res[mid2], i);
                cin >> med;
                if (med == i)
                {
                    if (mid2 - mid1 == 1)
                    {
                        lo = mid2;
                        break;
                    }
                    else
                    {
                        lo = mid1;
                        hi = mid2;
                    }
                }
                else if (med == res[mid2])
                {
                    if (hi - mid2 == 0)
                    {
                        lo = hi + 1;
                        break;
                    }
                    else
                    {
                        lo = mid2;
                    }
                }
                else if (med == res[mid1])
                {
                    if (mid1 - lo == 0)
                    {
                        break;
                    }
                    else
                    {
                        hi = mid1;
                    }
                }
            }
            res.insert(res.begin() + lo, i);
        }
        cout << makeString(res) << endl;
        flush(cout);
        cin >> L;
        // cin.ignore(numeric_limits<streamsize>::max(), '\n');
        if (L == -1)
        {
            exit(1);
        }
    }
}