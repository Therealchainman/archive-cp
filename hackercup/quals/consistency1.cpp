#include <bits/stdc++.h>
using namespace std;



int main() {
    int T;
    // freopen("inputs/consistency_chapter_1_validation_input.txt", "r", stdin);
    // freopen("outputs/consistency_chapter_1_validation_output.txt", "w", stdout);
    freopen("inputs/consistency_chapter_1_input.txt", "r", stdin);
    freopen("outputs/consistency_chapter_1_output.txt", "w", stdout);
    cin>>T;
    for (int t = 1;t<=T;t++) {
        string S;
        cin>>S;
        int n = S.size();
        int  nvowels = 0, nconsonants = 0, mxvowels=0, mxcons = 0;
        vector<int> counts(128,0);
        unordered_set<char> vowels = {'A', 'E', 'I', 'O', 'U'};
        for (char &c : S) {
            counts[c]++;
            if (vowels.count(c)==0) {
                nconsonants++;
                mxcons = max(counts[c], mxcons);
            } else {
                nvowels++;
                mxvowels = max(mxvowels, counts[c]);
            }
        }
        int ans = min(nvowels+2*(nconsonants-mxcons),nconsonants+2*(nvowels-mxvowels)); 
        printf("Case #%d: %d\n", t, ans);
    }
}