#include <bits/stdc++.h>
using namespace std;
/*
A line sweep algorithm can solve this with a map to track the used, unused, and available.
This way I can quickly update the current sum based by removing the smallest from used, and the largest from unused

*/
int main() {
    long long T, D, N, K;
    long long h, s, e;
    cin>>T;
    for (int t=1;t<=T;t++) {
        cin>>D>>N>>K;
        vector<vector<long long>> events;
        for (int i = 0;i<N;i++) {
            cin>>h>>s>>e;
            events.push_back({s,1,h});
            events.push_back({e+1,-1,h});
        }
        sort(events.begin(), events.end());
        map<long long,long long, greater<long long>> avail;
        long long ans = 0, delta;
        for (auto &event : events) {
            long long happy = 0, sz = 0;
            delta = event[1], h = event[2];
            if (delta==1) {
                avail[h]++;
            } else {
                avail[h]--;
                if (avail[h]==0) {
                    avail.erase(h);
                }
            }

            for (auto &[key, cnt]: avail) {
                happy += (min(K-sz, cnt)*key);
                sz += min(K-sz, cnt);
            } 
            ans = max(ans, happy);
        }

        printf("Case #%d: %lld\n", t, ans);
    }
}

// int maxHappiness = 0, happy = 0, delta, sz=0; // cnt => the number of current available attractions
// map<int,int> avail, used, unused;
// for (auto &event : events) {
//     delta = event[1], h = event[2];
//     if (delta==1) {
//         avail[h]++;
//         used[h]++;
//         happy+=h;
//         sz++;
//     } else {
//         avail[h]--;
//         if (avail[h]==0) {
//             avail.erase(h);
//         }
//         if (used.count(h)>0) {
//             used[h]--;
//             if (used[h]==0) {
//                 used.erase(h);
//             }
//             happy-=h;
//             sz--;
//         }
//     }
//     if (sz>K) {  
//         auto it = used.begin();
//         int ph = it->first;
//         used[ph]--;
//         if (used[ph]==0) {
//             used.erase(ph);
//         }
//         unused[ph]++;  // put into the unused map
//         sz--;
//         happy-=ph;
//     } else if(sz<K && unused.size()>0) {
//         auto it = unused.end();
//         it--;
//         int ph = it->first;
//         used[ph]++;
//         unused[ph]--;
//         if (unused[ph]==0) {
//             unused.erase(ph);
//         }
//         happy+=ph;
//         sz++;
//     }
//     maxHappiness = max(maxHappiness, happy);
// }
