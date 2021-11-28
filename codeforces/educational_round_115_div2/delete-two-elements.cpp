#include <bits/stdc++.h>
using namespace std;
/*
For this problem I found the arithmetic mean and then created a target value, checked that the target
value would give me ints because the array only contains ints.  Then binary search through the array, but
taking note of what we need to correct if index is in the overlap it doesn't count.  divide by 2 at the end because
we count each pair of indices twice.  
*/

long long binarySearch(vector<long long>& arr, long long target) {
    return upper_bound(arr.begin(), arr.end(), target)-arr.begin()-1;
}
int main() {
    long long T, N, a;
    string input;
    cin>>T;
    while (T--) {
        cin>>N;
        vector<long long> arr;
        for (long long i = 0;i<N;i++) {
            cin>>a;
            arr.push_back(a);
        }
        sort(arr.begin(), arr.end());
        long long sum = accumulate(arr.begin(), arr.end(), 0LL);
        double kk = (double)sum/(double)N;
        double ttarget = kk*(N-2);
        if (ttarget != (long long)ttarget) {
            cout<<0<<endl;
            continue;
        }
        long long target = (long long)ttarget;
        long long cnt = 0;
        for (long long i = 0;i<N;i++) {
            long long j = binarySearch(arr, sum-target-arr[i]), w = binarySearch(arr, sum-target-1-arr[i]);
            if (arr[j]!=sum-target-arr[i]) {continue;}
            cnt += j-w-(i>w && i<=j);
        }
        cout<<cnt/2<<endl;
    }
}

/*
1
7
1 2 3 4 5 6 7
*/