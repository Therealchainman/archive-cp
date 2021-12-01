#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <vector>
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

int countYes(vector<string> A) {
    int ans = 0;
    int count[26];
    for (int i = 0;i<26;i++) {
        count[i]=0;
    }
    for (string str : A) {
        for (char x : str) {
            count[x-'a']++;
        }
    }
    for (int i = 0;i<26;i++) {
        if (count[i]==A.size()) {
            ans++;
        }
    }
    return ans;
}

int main() {
    int res = 0;
    freopen("inputDay6.txt","r",stdin);
    string x;
    vector<string> A;
    while (getline(cin,x)) {
        if (x=="") {
            res+=countYes(A);
            A = {};
        } else {
            A.push_back(x);
        }
    }
    res+=countYes(A);
    cout<<res<<endl;
    return 0;
}