#include <bits/stdc++.h>
using namespace std;
const int N = 256;
long long days[9], pdays[9];
int main() {
    freopen("inputs/input.txt", "r", stdin);
    memset(days, 0, sizeof(days));
    string input, tmp;
    cin>>input;
    stringstream ss(input);
    while (getline(ss, tmp, ',')) {
        pdays[stoll(tmp)]++;
    }
    for (int day = 0;day<N;day++) {
        memset(days, 0, sizeof(days));
        for (int i = 0;i<8;i++) {
            days[i] += pdays[i+1];
        }
        days[6] += pdays[0];
        days[8] += pdays[0];
        for (int i = 0;i<9;i++) {
            pdays[i] = days[i];
        }
    }
    long long cnt = accumulate(pdays, pdays+9, 0LL);
    cout<<cnt<<endl;
}