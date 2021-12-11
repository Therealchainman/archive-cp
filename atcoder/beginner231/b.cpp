#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int N;
  string tmp;
  unordered_map<string, int> counts;
  int mx = 0;
  string winner;
  cin>>N;
  while (N--) {
    cin>>tmp;
    counts[tmp]++;
    if (counts[tmp]>mx) {
      mx = counts[tmp];
      winner=tmp;
    }
  }
  cout<<winner<<endl;
}