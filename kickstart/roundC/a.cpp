#include <bits/stdc++.h>
using namespace std;

int main()
{
    int T, K, N;
    cin >> T;
    string s;
    for (int t = 1; t <= T; t++)
    {
        long long ans = 0;
        cin >> N >> K;
        cin >> s;
        int n = s.size();
        int mid = (n + 1) / 2;
        for (int i = 0; i < mid; i++)
        {
            ans *= K;
            ans += s[i] - 'a';
        }
        if (mid > 0 && s[mid] > s[mid - 2])
        {
            ans++;
        }
        cout << "Case #" << t << ": " << ans << endl;
    }
}
