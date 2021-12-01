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
#define mx INT32_MAX
#define mn INT32_MIN
#define endl '\n'
using namespace std;

ll mod (ll a, ll b) {
    return (a % b + b) %b;
}

map<ll,ll> memMap;
map<ll,ll> results;

ll sum(map<ll,ll> mp) {
    ll res = 0;
    map<ll,ll>::iterator it;
    for (it=mp.begin();it!=mp.end();it++) {
        res+=(it->second);
    }
    return res;
}

string decBin(ll decimal) {
    ll mask;
    string ans = "";
    for (ll i = 0;i<36;i++) {
        mask = 1LL << i;
        if ((decimal&mask) > 0) {
            ans+='1';
        } else {
            ans+='0';
        }
    }
    return ans;
}

string evaluate(string mask, string bits) {
    int n = mask.size();
    for (int i = 0;i<n;i++) {
        if (mask[i]=='X') {
            bits[i]='X';
        }
        if (mask[i]=='1') {
            bits[i]='1';
        }
    }
    return bits;
}

ll binDec(string binary) {
    ll ans = 0;
    for (int i = 0;i<36;i++) {
        ans +=((binary[i]-'0')*pow(2,i));
    }
    return ans;
}

void updateMap(ll value, string bits) {
    int countX = 0;
    for (char ch : bits) {
        if (ch=='X') {
            countX++;
        }
    }
    for (ll mask = 0;mask<(1LL<<countX);mask++) {
        ll k = 0;
        string tmpKey = "";
        for (int i = 0;i<36;i++) {
            if (bits[i]=='X') {
                if ((mask&(1LL<<k))>0) {
                    tmpKey+='1';
                } else {
                    tmpKey+='0';
                }
                k++;
            } else {
                tmpKey+=bits[i];
            }
        }
        ll newKey = binDec(tmpKey);
        cout<<newKey<<endl;
        results[newKey]=value;
    }
}

void solve(string mask, vector<ll> memories) {
    if (mask == "") {
        return;
    }
    for (ll key : memories) {
        string bitsKey = decBin(key);
        string res = evaluate(mask, bitsKey);
        updateMap(memMap[key],res);
    }
}

int main() {

    freopen("inputDay14.txt","r",stdin);
    string input,tmp;
    string mask = "";
    vector<ll> memories;
    while (getline(cin,input)) {
        istringstream s(input);
        vector<string> inputs;
        while (getline(s,tmp,' ')) {
            inputs.push_back(tmp);
        }
        if (inputs[0]=="mask") {
            solve(mask, memories);
            mask = inputs[2];
            reverse(mask.begin(),mask.end());
            memories.clear();
        } else {
            int n = inputs[0].size();
            string key;
            bool start = false;
            for (int i = 0;i<n;i++) {
                if (inputs[0][i]==']') {
                    continue;
                }
                if (inputs[0][i]=='[') {
                    start = true;
                } else if (start) {
                    key+=inputs[0][i];
                }
            }
            ll intKey = stoi(key);
            ll intVal = stoi(inputs[2]);
            memories.push_back(intKey);
            memMap[intKey]=intVal;
        }
    }
    solve(mask,memories);
    cout<<sum(results)<<endl;
    return 0;
}
