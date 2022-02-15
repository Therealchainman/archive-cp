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
 
*/
template <class T>
string makeString(vector<T> &vec)
{
    stringstream res;
    copy(vec.begin(), vec.end(), ostream_iterator<T>(res, " "));
    return res.str().c_str();
}
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, N;
    cin >> T;
    string s;
    for (int t = 1; t <= T; t++)
    {
        cin >> N;
        cin >> s;
        vector<int> vec(N, 0);
        int cnt = 1;
        vec[0] = cnt;
        for (int i = 1; i < N; i++)
        {
            if (s[i] > s[i - 1])
            {
                cnt++;
            }
            else
            {
                cnt = 1;
            }
            vec[i] = cnt;
        }
        cout << "Case #" << t << ": " << makeString(vec) << endl;
    }
}