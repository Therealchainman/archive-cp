#include <bits/stdc++.h>
using namespace std;

int main()
{
    int T;
    long long G;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        int cnt = 1;
        cin >> G;
        G *= 2;
        for (long long n = 2; G / n > (n - 1); n++)
        {
            if (G % n == 0)
            {
                long long K = G / n - (n - 1);

                if (K % 2 == 0)
                {
                    cnt++;
                }
            }
        }
        cout << "Case #" << t << ": " << cnt << endl;
    }
}
