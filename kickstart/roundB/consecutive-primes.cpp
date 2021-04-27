
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
// Prints vector of pairs
template <class T1, class T2>
void printPairVec(vector<pair<T1, T2>> &p)
{
    cout << "[";
    for (auto pp : p)
    {
        cout << "(" << pp.first << "," << pp.second << "),";
    }
    cout << "]" << endl;
}

/*
    Binary search type of problem.  The tricky part for me was finding a way to compute the prime numbers
    So I computed the next prime number from the current mid integer, and then the next prime number after that.  
    if this mid was too small.

    I learned an algorithm called miller_rabin for checking if a number is prime.  
*/
template <class T>
string makeString(vector<T> &vec)
{
    stringstream res;
    copy(vec.begin(), vec.end(), ostream_iterator<T>(res, " "));
    return res.str().c_str();
}

// ll miller_rabin(int n)
// {
// }

ll isPrime(int n)
{
    if (n == 2)
    {
        return true;
    }
    if (n <= 1 || n % 2 == 0)
    {
        return false;
    }
    for (int i = 3; (i * i) <= n; i += 2)
    {
        if (n % i == 0)
        {
            return false;
        }
    }
    return true;
}

ll findPrime(int n)
{
    while (!isPrime(n))
    {
        n++;
    }
    return n;
}

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T;
    ll Z;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> Z;
        int lo = 0, hi = 1e9;
        ll p, q;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) >> 1;
            p = findPrime(mid);
            q = findPrime(p + 1);
            if (p * q <= Z)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }
        ll ans = lo * findPrime(lo + 1);
        cout << "Case #" << t << ": " << ans << endl;
    }
}
