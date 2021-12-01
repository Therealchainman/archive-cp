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

int computeAcc(vector<pair<string,int>> actions, int swapIdx) {
    int ans = 0;
    int i = 0;
    set<int> vis;
    while (vis.find(i)==vis.end()) {
        vis.insert(i);
        pair<string,int> p = actions[i];
        if (i == swapIdx) {
            if (p.first=="nop") {
                i+=p.second;
            } else {
                i++;
            }
            continue;
        }
        if (p.first=="acc") {
            ans+=p.second;
            i++;
        } else if (p.first=="nop") {
            i++;
        } else {
            i+=p.second;
        }
    }
    return ans;
}

// Compute the index for the appropriate swap idx that is basically 
bool isLastIdx(vector<pair<string,int>> actions, int idx) {
    int i = 0;
    set<int> vis;
    bool firstSeen = false;
    while (vis.find(i)==vis.end()) {
        if (i==actions.size()) {
            return true;
        }
        vis.insert(i);
        pair<string,int> p = actions[i];
        if (i==idx && !firstSeen) {
            firstSeen = true;
            if (p.first=="nop") {
                i+=p.second;
            } else {
                i++;
            }
            continue;
        }
        if (p.first=="acc" || p.first=="nop") {
            i++;
        } else {
            i+=p.second;
        }
    }
    return i==actions.size();
}

vector<int> swapIndices(vector<pair<string,int>> actions, string action) {
    int i = 0;
    set<int> vis;
    vector<int> ret(0,0);
    while (vis.find(i)==vis.end()) {
        vis.insert(i);
        pair<string,int> p = actions[i];
        if (p.first=="nop") {
            if (action=="nop") {
                ret.push_back(i);
            }
            i++;

        } else if (p.first=="jmp") {
            if (action=="jmp") {
                ret.push_back(i);
            }
            i+=p.second;
        } else {
            i++;
        }
    }
    return ret;
}

int solve(vector<pair<string,int>> A, bool swap) {
    if (!swap) {
        return computeAcc(A, -1);
    } else {
        vector<int> swapIdxToJmp = swapIndices(A, "nop");
        vector<int> swapIdxToNop = swapIndices(A, "jmp");
        for (int ji : swapIdxToJmp) {
            if (isLastIdx(A, ji)) {
                return computeAcc(A, ji);
            }
        }
        for (int ni : swapIdxToNop) {
            if (isLastIdx(A, ni)) {
                return computeAcc(A, ni);
            }
        }
    }
    return 0;
}


int main() { 

    freopen("inputDay8.txt","r",stdin);
    string input;
    vector<pair<string,int>> A;
    while (getline(cin,input)) {
        string key = "";
        string val = "";
        bool flag = true;
        for (int i = 0;i<input.size();i++) {
            if (input[i]==' ') {
                flag=false;
            }
            if (flag) {
                key+=input[i];
            } else {
                val+=input[i];
            }
        } 
        int x;
        if (val[0]=='-') {
            x = -stoi(val.substr(1));
        } else {
            x = stoi(val.substr(1));
        }
        A.push_back({key,x});
    }
    int part1 = solve(A, false);
    int part2 = solve(A, true);
    cout<<part1<<endl;
    cout<<part2<<endl;
    return 0;
}