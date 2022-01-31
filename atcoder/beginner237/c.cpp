#include <bits/stdc++.h>
using namespace std;

bool isPalindrome(string& S, int i, int j) {
    while (i<j) {
        if (S[i++]!=S[j--]) return false;
    }
    return true;
}

int main() {
    string S;
    cin>>S;
    int N = S.size(), i = 0, j = N-1;
    while (i<j) {
        if (S[i]=='a' && S[j]=='a') {
            i++;
            j--;
        } else if (S[j]=='a') {
            j--;
        } else {
            break;
        }
    }
    if (isPalindrome(S, i, j)) {
        cout << "Yes" << endl;
    } else {
        cout << "No" << endl;
    }
}