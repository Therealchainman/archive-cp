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

const string imp = "IMPOSSIBLE";

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

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, N, C;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> N >> C;
        if (C < N - 1 || C > N * (N + 1) / 2 - 1)
        {
            cout << "Case #" << t << ": " << imp << endl;
            continue;
        }
        int left = 0, right = N - 1, candCost;
        bool decLeft = false;
        vector<int> A;
        for (int i = 1; i <= N; i++)
        {
            A.push_back(i);
        }
        C -= N - 1;
        while (left < right && C > 0)
        {
            candCost = right - left;
            if (candCost <= C)
            {
                reverse(A.begin() + left, A.begin() + right + 1);
                C -= candCost;
            }
            if (decLeft)
            {
                left++;
                decLeft = false;
            }
            else
            {
                right--;
                decLeft = true;
            }
        }
        cout << "Case #" << t << ": " << makeString(A) << endl;
    }
}