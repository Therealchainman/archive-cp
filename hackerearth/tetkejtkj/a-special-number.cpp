#include <bits/stdc++.h>
using namespace std;
bool isDivFour(int a) {
	int s = 0;
	while (a>0) {
		s += (a%10);
		a/=10;
	}
	return s%4==0;
}
int main() {
	int T, a;
	cin>>T;
	while (T--) {
		cin>>a;
		int s = 0;
		while (!isDivFour(a)) {
			a++;
		}
		cout<<a<<endl;
	}
}

4
2 3 4
3 1 4
4 1 2
1 3 2