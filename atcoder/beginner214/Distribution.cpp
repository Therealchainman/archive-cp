#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, t;
    cin>>N;
    vector<int> S(N,0), ans(N,0);
    priority_queue<pair<int,int>,vector<pair<int,int>>, greater<pair<int,int>>> minHeap;
    for (int i = 0;i<N;i++) {
        cin>>S[i];
    }
    for (int i = 0;i<N;i++) {
        cin>>t;
        minHeap.emplace(t, i);
    }
    int cnt = 0;
    while (!minHeap.empty() && cnt<N) {
        int time, snuke;
        tie(time, snuke) = minHeap.top();
        minHeap.pop();
        if (ans[snuke]>0) {
            continue;
        }
        ans[snuke]=time;
        minHeap.emplace(time+S[snuke],(snuke+1)%N);
    }
    for (int i = 0;i<N;i++) {
        cout<<ans[i]<<endl;
    }
}