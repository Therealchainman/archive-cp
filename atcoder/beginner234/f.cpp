#include <bits/stdc++.h>
using namespace std;
const int MOD = 998244353;
long long fact[5001];
int freq[26];
long long power(long long a, long long b) {
    long long result = 1;
    while (b>0) {
        if (b%2 == 1) {
            result = (result*a)%MOD;
        }
        a = (a*a)%MOD;
        b/=2;
    }
    return result;
}
long long inverse(long long a, long long b) {
    return power(a, b-2);
}
long long divide(long long a, long long b) {
    return (a*inverse(b, MOD))%MOD;
}
int main() {
    string S;
    cin>>S;
    int n = S.size();
    fact[0] = 1;
    for (int i = 1;i<=n;i++) {
        fact[i] = (fact[i-1]*i)%MOD;
    }
    // keep visited with a string. 
    long long cnt = 0;
    unordered_set<string> visited;
    for (int i = 0;i<n;i++) {
        memset(freq,0,sizeof(freq));
        for (int j = i;j<n;j++) {
            freq[S[j]-'a']++;
            string serial = "";
            for (int k = 0;k<26;k++) {
                serial += to_string(freq[k]);
            }
            if (visited.count(serial)) continue;
            visited.insert(serial);
            long long num = fact[j-i+1];
            long long den = 1;
            for (int k = 0;k<26;k++) {
                den = (den*fact[freq[k]])%MOD;
            }
            long long cur = divide(num,den);
            cnt = (cnt+cur)%MOD;
        }
    }
    cout<<cnt<<endl;
}

/*
11 characters, h=2, y=2, g=3
number permutatios = 11!/(2!2!3!)= result.  
*/