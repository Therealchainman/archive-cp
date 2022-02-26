#include <bits/stdc++.h>
using namespace std;
 
int arr[10];
 
int main() {
  for (int i = 0;i<10;i++) {
  	cin>>arr[i];
  }
  auto itr = find_if(arr, arr+10, [&](const auto& a) {return a==0;}
  int index = *itr;
  for (int i = 0;i<3;i++) {
  	index = arr[index];
  }
  cout<<arr[index]<<endl;
}