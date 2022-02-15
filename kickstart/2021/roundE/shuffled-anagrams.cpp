#include <bits/stdc++.h>
using namespace std;
const string IMP = "IMPOSSIBLE";
/*
Shuffled anagrams is a problem that asks for you to generate an anagram A of string S such that 
S[i]!=A[i] for all i<S.size().  So at every index in the original string the anagram needs to have 
a different character.

I learned in this problem that next_permutation will only return the lexicographically greater permutations.
Unless you sort the string to be in the smallest lexicographic permutation you will not be able to 
generate all n! permutations of a string. 
*/


/*
This returns true if the shuffled anagram is possible
This is the first solution that uses next_permutation
*/
// string shuffledAnagram(string& s) {
//     int n = s.size();
//     string A = s;
//     bool found = true;
//     vector<int> counts(26,0);
//     for (char &c : s) {
//         counts[c-'a']++;
//     }
//     for (int i = 0;i<26;i++) {
//         if (counts[i]>n/2) {
//             return IMP;
//         }
//     }
//     sort(A.begin(),A.end());
//     do {
//         found = true;
//         for (int i = 0;i<n;i++) {
//             if (s[i]==A[i]) {
//                 found = false;
//                 break;
//             }
//         }
//         if (found) {
//             break;
//         }
//     } while (next_permutation(A.begin(),A.end()));
//     return A;
// }
// int main() {
//     int T;
//     // freopen("in.txt","r",stdin);
//     // freopen("out.txt", "w", stdout);
//     cin>>T;
//     for (int t=1;t<=T;t++) {
//         string s;
//         cin>>s;
//         string res = shuffledAnagram(s);
//         printf("Case #%d: %s\n", t, res.c_str());
//     }
// }


/*
This is the optimized solution that uses the splitting it in half
and sorting the string and then swapping.  
*/
string shuffledAnagram(string& s) {
    int n = s.size();
    unordered_map<char,vector<int>> charIndex;
    vector<int> counts(26,0);
    for (int i = 0;i<n;i++) {
        counts[s[i]-'a']++;
        charIndex[s[i]].push_back(i);
    }
    for (int i = 0;i<26;i++) {
        if (counts[i]>n/2) {
            return IMP;
        }
    }
    sort(s.begin(),s.end());
    vector<int> pos(n,0); // Array stores index position of each element, so can map
    // an element back to original order.
    for (int i = 0;i<n;i++) {
        pos[i]=charIndex[s[i]].back();
        charIndex[s[i]].pop_back();
    }
    for (int i = 0;i<n/2;i++) {
        swap(s[i], s[i+n/2]);
    }
    if (n%2!=0) {
        swap(s[n/2], s[n-1]);
    }
    string A = s;
    for (int i = 0;i<n;i++) {
        A[pos[i]] = s[i];
    }
    return A;
}
int main() {
    int T;
    cin>>T;
    for (int t=1;t<=T;t++) {
        string s;
        cin>>s;
        string res = shuffledAnagram(s);
        printf("Case #%d: %s\n", t, res.c_str());
    }
}