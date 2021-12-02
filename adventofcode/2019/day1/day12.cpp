#include <bits/stdc++.h>
using namespace std;

int main() {
    freopen("inputs/input.txt", "r", stdin);
    string tmp;
    int mass, sum = 0;
    while (getline(cin, tmp)) {
        mass = stoi(tmp);
        mass = (mass/3)-2;
        while (mass>0) {
            sum += mass;
            mass = (mass/3)-2;
        }
    }
    cout<<sum<<endl;
}