#include <bits/stdc++.h>
using namespace std;
/*
part 1
This could be a good problem to implement a doubly linked list that way we can remove from ratingArr in O(1) time, 
and not worry about other things.
*/
int convBinaryToDecimal(string& s) {
    int n = s.size(), dec = 0;
    for (int i = 0; i < n; i++) {
        dec += ((s[i] - '0')*(1<<(n-i-1)));
    }
    return dec;
}
int getRating(int index, vector<string>& ratingArr, int indicator) {
    if (ratingArr.size()==1) {
        return convBinaryToDecimal(ratingArr[0]);
    }
    int n = ratingArr[0].size();
    if (index == ratingArr[0].size()) {
        return -1;
    }
    vector<string> arr;
    for (int i = 0; i < n; i++) {
        string s = "";
        for (int j = 0; j < ratingArr.size(); j++) {
            s += ratingArr[j][i];
        }
        arr.push_back(s);
    }
    int cnt1 = count(arr[index].begin(), arr[index].end(), '1'), cnt0 = count(arr[index].begin(), arr[index].end(), '0');
    vector<string> tmp;
    for (int i = 0;i<ratingArr.size();i++) {
        if (indicator) {
            char interest = (cnt1>=cnt0)?'1':'0';
            if (ratingArr[i][index] == interest) {
                tmp.push_back(ratingArr[i]);
            }
        } else {
            char interest = (cnt1<cnt0) ? '1' : '0';
            if (ratingArr[i][index] == interest) {
                tmp.push_back(ratingArr[i]);
            }
        }
    }
    return getRating(index + 1, tmp, indicator);
}
int main() {
    freopen("inputs/input.txt", "r", stdin);
    vector<string> binaryArr;
    string input;
    while (getline(cin, input)) {
        binaryArr.push_back(input);
    }
    cout << getRating(0, binaryArr, 1)*getRating(0, binaryArr, 0) << endl;
}
