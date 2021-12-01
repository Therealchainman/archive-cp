#include <iostream>
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
using namespace std::chrono;

ll mod (ll a, ll b) {
    return (a % b + b) %b;
}


int solve(vector<int> nums, int end) {
    int n = nums.size();
    unordered_map<int,int> posNum;
    int curNum;
    int i;
    for (i = 0;i<n;i++) {
        posNum[nums[i]]=i;
        curNum=nums[i];
    }
    int nextNum = 0;
    int count[10];
    while (i<end) {
        curNum=nextNum;
        if (posNum.find(nextNum)==posNum.end()) {\
            nextNum = 0;
            posNum[curNum]=i;
        } else {
            int prevIndex = posNum[curNum];
            int curIndex = i;
            posNum[curNum]=curIndex;
            nextNum=curIndex-prevIndex;
        }
        i++;
    }
    return curNum;
}

int main() {

    freopen("big.txt","r",stdin);
    string input,tmp;
    vector<int> starting_nums;
    while (getline(cin,input)) {
        istringstream s(input);
        vector<string> inputs;
        while (getline(s,tmp,',')) {
            int x = stoi(tmp);
            starting_nums.push_back(x);
        }
    }
    auto start = high_resolution_clock::now();
    int res = solve(starting_nums,30000000);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop-start);
    cout<<duration.count()<<endl;
    cout<<res<<endl;
    return 0;
}
