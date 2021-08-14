#include <bits/stdc++.h>
using namespace std;

int main() {
    int T, N;
    cin>>T;
    while (T--) {
        cin>>N;
        string n = to_string(N);
        int ans = 0;
        for (int i = 0;i<n.size();i++) {
            ans = max(ans, n[i]-'0');
        }
        printf("%d\n",ans);
    }
}