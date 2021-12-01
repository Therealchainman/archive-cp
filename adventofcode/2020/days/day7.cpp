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

// counts the number of bags that can hold a bag of type bag
int countBags(map<string,map<string,int>> graph, string bag) {
    int count = 0;
    if (graph[bag].empty()) {
        return 0;
    } else {
        for (pair<string,int> pai : graph[bag]) {
            count += pai.second*(1+countBags(graph, pai.first));
        }
    }
    return count;
}


int main() { 
    int res = 0;
    string input;
    freopen("inputDay7.txt","r",stdin);
    map<string, map<string,int>> graph;
    vector<string> sent;
    while (getline(cin,input)) {
        istringstream s(input);
        string tmp, a,key,val;
        string int_num = "^0$|^[1-9][0-9]*$";
        regex pattern(int_num);
        key="";
        sent = {};
        int cnt;
        map<string,int> listOfBags;
        while (getline(s,tmp,' ')) {
            if (regex_match(tmp,pattern)) {
                cnt=stoi(tmp);
                continue;
            } else if (tmp=="contain") {
                continue;
            } else if (tmp=="no") {
                break;
            }
            sent.push_back(tmp);
            if (sent.size()==3) {
                for (int i = 0;i<2;i++) {
                    key+=sent[i];
                    key+=' ';
                }
            } else if (sent.size()%3==0) {
                val = "";
                for (int i = sent.size()-3;i<sent.size()-1;i++) {
                    val+=sent[i];
                    val+=' ';
                }
                listOfBags[val]=cnt;
            }
        }
        graph[key]=listOfBags;
    }
    string curBag= "shiny gold ";
    res = countBags(graph, curBag);
    cout<<res<<endl;

    return 0;
}