#include <bits/stdc++.h>
using namespace std;

multiset<long long> A;

long long klargest(long long x, int k) {
  auto itr = A.upper_bound(x);
  for (int i = 0;i<k;i++) {
    if (itr == A.begin()) return -1;
    itr = prev(itr);
  }
  return *itr;
}

long long ksmallest(long long x, int k) {
  auto itr = A.lower_bound(x);
  if (itr == A.end()) return -1;
  for (int i = 1;i<k;i++) {
    itr = next(itr);
    if (itr==A.end()) return -1;
  }
  return *itr;
}

int main() {
  int Q, t, k;
  long long x;
  freopen("input.txt", "r", stdin);
  cin>>Q;
  for (int i = 0;i<Q;i++) {
    cin>>t>>x;
    if (t==1) {
      A.insert(x);
    } else if (t==2) {
      cin>>k;
      cout<<klargest(x, k)<<endl;
    } else {
      cin>>k;
      cout<<ksmallest(x,k)<<endl;
    }
  }
}