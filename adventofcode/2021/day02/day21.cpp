#include <bits/stdc++.h>
using namespace std;
/*
part 1
*/
int main() {
    freopen("inputs/input1.txt", "r", stdin);
    freopen("outputs/output1.txt", "w", stdout);
    string input;
    long long depth = 0, hor = 0;
    while (getline(cin, input)) {
        int pos = input.find(" ");
        string direction = input.substr(0, pos);
        int magnitude = stoi(input.substr(pos + 1));
        if (direction == "down") {
            depth += magnitude;
        } else if (direction == "up") {
            depth -= magnitude;
        } else if (direction == "forward") {
            hor += magnitude;
        }
    }
    cout<<depth*hor<<endl;
}
