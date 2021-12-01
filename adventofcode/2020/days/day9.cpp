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
#define endl '\n'
using namespace std;

bool check(vector<ll> prev, ll val, ll start) {
    set<ll> vis(prev.begin()+start,prev.end());
    for (int i = start;i<start+prev.size();i++) {
        if (vis.find(val-prev[i])!=vis.end() && prev[i]!=2*prev[i]) {
            return true;
        }
    }
    return false;
}

ll solve(vector<ll> arr, ll sz) {
    vector<ll> prev;
    for (int i = 0;i<sz;i++) {
        prev.push_back(arr[i]);
    }
    ll start = 0;
    for (int j = sz;j<arr.size();j++) {
        if (!check(prev, arr[j], start++)) {
            return arr[j];
        }
        prev.push_back(arr[j]);
    }
    return 0;
}

ll solve2(vector<ll> arr, ll goal) {
    ll curMin = INT64_MAX;
    ll curMax = INT64_MIN;
    ll sum = 0;
    set<ll> res;
    ll lo = 0, hi = 0;
    while (hi<arr.size() && sum!=goal) {
        sum += arr[hi];
        res.insert(arr[hi++]);
        while (lo<arr.size() && sum>goal) {
            sum-=arr[lo];
            res.erase(arr[lo++]);
        }
    }
    set<ll>::iterator it;
    for (it=res.begin();it!=res.end();it++) {
        curMin = min(curMin, *it);
        curMax = max(curMax, *it);
    }
    return curMin+curMax;
}

int main() { 
    freopen("inputDay9.txt","r",stdin);
    ll x;
    vector<ll> arr;
    while (cin >>x) {
        arr.push_back(x);
    }
    ll preAmbleSize = 25;
    ll res = solve(arr,preAmbleSize);
    cout<<solve2(arr, res)<<endl;
    return 0;
}