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
        int prev = s[0]-'0', cnt = 1;
        for (int j = 1;j<s.size();j++) {
            cnt += (s[j]-'0' == prev);
            if (s[j]-'0' != prev || j==s.size()-1) {
                adj |= (cnt==2);
                cnt = 1;
                prev = s[j]-'0';
            }
            if (s[j]<s[j-1]) {
                nonDecreasing = false;
            }
        }
        count += adj && nonDecreasing;
    }
    cout<<count<<endl;
}