#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

int main() {
    int t = read();
	while (t--) {
		int n = read();
		vector<int> arr(n);
		for (int i = 0; i < n; i++) {
			int x = read();
			arr[i] = x;
		}
		int left = 0, right = n - 1;
		int smallest = 1, largest = n;
		while (left < right) {
			if (arr[left] == smallest) {
				left++;
				smallest++;
			} else if (arr[right] == largest) {
				right--;
				largest--;
			} else if (arr[left] == largest) {
				left++;
				largest--;
			} else if (arr[right] == smallest) {
				right--;
				smallest++;
			} else {
				break;
			}
		}
		if (left == right) {
			cout << -1 << endl;
		} else {
			cout << left + 1 << " " << right + 1 << endl;
		}
	}
    return 0;
}