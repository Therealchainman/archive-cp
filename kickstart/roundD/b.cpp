#include <bits/stdc++.h>
using namespace std;

int main() {
    int T, N;
    long long C, l, r;
    cin>>T;
    for (int t=1;t<=T;t++) {
        vector<pair<long long, int>> events;
        cin>>N>>C;
        for (int j=0;j<N;j++) {
            cin>>l>>r;
            if (l+1==r) {
                continue;
            }
            events.emplace_back(l+1, -1);
            events.emplace_back(r,1);
        }
        sort(events.begin(), events.end());
        priority_queue<vector<long long>> counts; // (count)
        int prev = 0;
        long long edge, prevEdge=events[0].first;
        int delta, i=0;
        while (i<events.size()) {
            tie(edge, delta) = events[i];
            delta=-delta;
            while (i<events.size() && edge==prevEdge) {
                tie(edge, delta);
                prev+=delta;
                i++;
            }
            if (prev>0) {
                counts.push({prev, prevEdge, edge-1});
            }
            prevEdge = edge;
        }
        long long ans = N, cuts = 0, cnt;
        while (!counts.empty() && cuts<C) {
            auto here = counts.top();
            cnt=here[0], prevEdge=here[1], edge=here[2];
            printf("cnt=%lld, prevEdge=%lld, edge=%lld\n", cnt, prevEdge, edge);
            long long cur = min(C-cuts, edge-prevEdge); // num cuts made
            ans = (ans+cur*cnt);
            counts.pop();
            cuts+=cur;
        }
        printf("Case #%d: %lld\n", t, ans);
    }
}