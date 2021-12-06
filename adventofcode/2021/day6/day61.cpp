#include <bits/stdc++.h>
using namespace std;
const int N = 80;
long long lanternFish[9];
int main() {
    freopen("inputs/input.txt", "r", stdin);
    memset(lanternFish, 0, sizeof(lanternFish));
    string input, tmp;
    cin>>input;
    stringstream ss(input);
    while (getline(ss, tmp, ',')) {
        lanternFish[stoll(tmp)]++;
    }
    for (int day = 0;day<N;day++) {
        long long born = lanternFish[0];
        for (int i = 0;i<8;i++) {
            lanternFish[i] = lanternFish[i+1];
        }
        lanternFish[6] += born;
        lanternFish[8] = born;
    }
    long long cnt = accumulate(lanternFish, lanternFish+9, 0LL);
    cout<<cnt<<endl;
}