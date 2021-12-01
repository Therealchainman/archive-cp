#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <vector>
#include <regex>
#include <set>
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
#define rep(i,n) for (i = 0; i < n; ++i) 
#define REP(i,k,n) for (i = k; i <= n; ++i) 
#define REPR(i,k,n) for (i = k; i >= n; --i)
#define pb push_back
#define all(a) a.begin(), a.end()
#define fastio ios::sync_with_stdio(0); cin.tie(0); cout.tie(0)
#define ll long long 
#define uint unsigned long long
#define inf 0x3f3f3f3f3f3f3f3f
#define mxl INT64_MAX
#define mnl INT64_MIN
#define mx INT_MAX
#define mn INT_MIN
#define endl '\n'
using namespace std;

ll mod (ll a, ll b) {
    return (a % b + b) %b;
}

ll chineseRemainderTheoremNaive(vector<ll> times) {
    int k = times.size();
        for (ll i = times[0];;i+=times[0]) {
            int offset = 1;
            int j;
            for (j=1;j<k;j++) {
                ll bus = times[j];
                if (bus==-1) {
                    offset++;
                    continue;
                }
                if (i%bus!=bus-offset) {
                    break;
                }
                offset++;
            }
        if (j==k) {
            return i;
        }
    }
    return 0;
}

ll solve(vector<vector<ll>> times) {
    int k = times.size();
    int si = 1;
    ll stepSize = times[0][0];
    ll i = 0;
    while (si<k) {
        while (true) {
            i+=stepSize;
            if ((i+times[si][1])%times[si][0] == 0) {
                break;
            }
        }
        stepSize*=times[si][0];
        si++;
    }
    return i;
}

int main() {

    freopen("inputDay13.txt","r",stdin);
    string buses;
    cin >>buses;
    istringstream s(buses);
    string bus;
    vector<vector<ll>> times;
    int offset = -1;

    while (getline(s,bus,',')) {
        offset++;
        if (bus=="x") {
            continue;
        } 
        ll val = stoi(bus);
        times.push_back({val,offset});
    }
    cout<<solve(times)<<endl;
    return 0;
}