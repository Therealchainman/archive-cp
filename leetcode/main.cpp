#include <bits/stdc++.h>
using namespace std;
#define x first
#define y second

class Solution {
public:
    long long validSubstringCount(string word1, string word2) {
        int N = word1.size();
        vector<long long> freq(26, 0);
        long long need = 0;
        for (char c : word2) {
            if (freq[c - 'a'] == 0) {
                need++;
            }
            freq[c - 'a']++;
        }
        long long ans = 0;
        for (int l = 0, r = 0; r < N; r++) {
            int v = word1[r] - 'a';
            freq[v]--;
            if (freq[v] == 0) {
                need--;
            }
            while (need == 0) {
                int u = word1[l] - 'a';
                if (freq[u] == 0) break;
                freq[u]++;
                l++;
            }
            if (need == 0) {
                ans += l + 1;
            }
        }
        return ans;
    }
};