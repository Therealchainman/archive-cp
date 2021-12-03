#include <bits/stdc++.h>
using namespace std;

int main() {
    freopen("inputs/input.txt", "r", stdin);
    string input;
    cin>>input;
    int pos = input.find("-");
    int start = stoi(input.substr(0, pos));
    int end = stoi(input.substr(pos+1));
    int count = 0;
    for (int i = start;i<=end;i++) {
        bool adj = false, nonDecreasing = true;
        string s = to_string(i);
        for (int j = 1;j<s.size();j++) {
            if (s[j]==s[j-1]) {
                adj = true;
            } 
            if (s[j]<s[j-1]) {
                nonDecreasing = false;
            }
        }
        count += adj && nonDecreasing;
    }
    cout<<count<<endl;
}