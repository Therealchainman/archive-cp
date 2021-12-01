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

ll mod (ll a, ll b) {
    return (a % b + b) %b;
}

map<char, pair<int,int>> dirs = {{'E', {1,0}}, {'S',{0,-1}},{'W',{-1,0}},{'N',{0,1}}};

ll wx = 10, wy = 1;

void updateWayPoint(ll delta, char d) {
        ll dx, dy;
        tie(dx,dy) = dirs[d];
        wx+=(dx*delta);
        wy+=(dy*delta);
}

void rotation(ll delta, char orientation) {
    ll i = mod(delta/90,4);
    ll nx, ny;
    int R, L;
    R = orientation == 'R' ? 1 : 0;
    L = orientation == 'L' ? 1 : 0;
    while (i>0) {
        nx = R*wy+-L*wy;
        ny = -R*wx+L*wx;
        wx = nx;
        wy = ny;
        i--;
    }
}

int main() {

    freopen("inputDay12.txt","r",stdin);
    ll x = 0, y = 0;
    string direction;
    while (cin>>direction) {
        char d = direction[0];
        ll delta = stoi(direction.substr(1));
        if (d == 'F') {
            x+=(wx*delta);
            y+=(wy*delta);
        } else if (d == 'R' || d == 'L') {
            rotation(delta, d); 
        } else {
            updateWayPoint(delta, d);
        }
    }
    ll res = abs(x) + abs(y);
    cout<<res<<endl;
    return 0;
}