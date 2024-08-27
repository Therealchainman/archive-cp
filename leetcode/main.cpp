#include <bits/stdc++.h>
using namespace std;


class Solution {
public:
    int countPairs(vector<int>& nums) {
        unordered_map<int, int> freq;
        int N = nums.size();
        int ans = 0;
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                string si = to_string(nums[i]);
                string sj = to_string(nums[j]);
                int ham = 0;
                if (si.size() < sj.size()) {
                    int delta = sj.size() - si.size();
                    si = string(delta, '0') + si;
                } else if (si.size() > sj.size()) {
                    int delta = si.size() - sj.size();
                    sj = string(delta, '0') + sj;
                }
                bool ok = false;
                for (int k = 0; k < si.size() && !ok; k++) {
                    for (int l = 0; l < sj.size() && !ok; l++) {
                        swap(si[k], si[l]);
                        for (int r = 0; r < si.size(); r++) {
                            if (si[r] != sj[r]) ham++;
                        }
                        if (ham <= 2) {
                            ok = true;
                        }
                        swap(si[k], si[l]);
                    }
                }
                ans += ok;
            }
        }
        return ans;
    }
};

