#include <bits/stdc++.h>
using namespace std;
/*
Simulation of the operations naively. 
Solve in O(n) with 10 buckets and saving the locations and using a bidirectional linked list for O(1) removal through the locations
optimized with an array of ints.
*/

const int n = 5e5+5;
int nxt[n],prv[n];

int ctoi(char c){
    return c-'0';
}

char itoc(int i){
    return i+'0';
}

int main() {
    int T, N;
    string S;
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    cin>>T;
    for (int t=1;t<=T;t++) {
        cin>>N>>S;
        for (int i = 0;i<=N;i++) {
            nxt[i] = i+1;
            prv[i+1] = i;
        }
        unordered_set<int> buckets[10];
        int cnt = 0;
        for (int i = 0;i<N-1;i++) {
            if (ctoi(S[i+1]) == ctoi(S[i]+1)%10) {
                buckets[ctoi(S[i])].insert(i+1);
                cnt++;
            }
        }
        int i = 0;
        while (cnt>0) {
            while (buckets[i].empty()) {
                i = (i+1)%10;
            }
            int index = *buckets[i].begin();
            buckets[i].erase(index);
            cnt--;
            S[index-1] = itoc((ctoi(S[index-1])+2)%10);
            int rem = nxt[index];
            assert(rem <= N);
            prv[nxt[rem]] = prv[rem];
            nxt[prv[rem]] = nxt[rem];
            for (int i = 0;i<10;i++) {
                if (buckets[i].count(rem)) {
                    buckets[i].erase(rem);
                    cnt--;
                }
            }
            for (int i = 0;i<10;i++) {
                if (buckets[i].count(prv[index])) {
                    buckets[i].erase(prv[index]);
                    cnt--;
                }
            }
            if (nxt[index]<=N && ctoi(S[nxt[index]-1])==(ctoi(S[index-1])+1)%10) {
                buckets[ctoi(S[index-1])].insert(index);
                cnt++;
            }
            if (prv[index]>0 && ctoi(S[index-1])==(ctoi(S[prv[index]-1])+1)%10) {
                buckets[ctoi(S[prv[index]-1])].insert(prv[index]);
                cnt++;
            }
        }
        string res = "";
        for (int i = 1;i<=N;i=nxt[i]) {
            res += S[i-1];
        }
        printf("Case #%d: %s\n", t, res.data());
    }
}