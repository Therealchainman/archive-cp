#include <bits/stdc++.h>
using namespace std;
/*
part 1
*/
int convBinaryToDecimal(string& s) {
    int n = s.size(), dec = 0;
    for (int i = 0; i < n; i++) {
        dec += ((s[i] - '0')*(1<<(n-i-1)));
    }
    return dec;
}
int main() {
    freopen("inputs/input.txt", "r", stdin);
    vector<string> binaryArr;
    string input;
    while (getline(cin, input)) {
        binaryArr.push_back(input);
    }
    int n = binaryArr[0].size();
    vector<string> arr;
    for (int i = 0;i<n;i++) {
        string s = "";
        for (int j = 0;j<binaryArr.size();j++) {
            s += binaryArr[j][i];
        }
        arr.push_back(s);
    }
    string gamm = "";
    for (int i = 0;i<n;i++) {
        gamm += (count(arr[i].begin(), arr[i].end(), '1') > count(arr[i].begin(), arr[i].end(), '0') ? '1' : '0');
    }
    int gamma = convBinaryToDecimal(gamm);
    int epsilon = gamma ^ ((1<<n)-1);
    cout << gamma*epsilon << endl;
}
