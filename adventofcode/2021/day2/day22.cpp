#include <bits/stdc++.h>
using namespace std;
/*
part 2
*/
int main() {
    freopen("inputs/input1.txt", "r", stdin);
    freopen("outputs/output2.txt", "w", stdout);
    string input;
    long long depth = 0, hor = 0, aim = 0;
    while (getline(cin, input)) {
        int pos = input.find(" ");
        string direction = input.substr(0, pos);
        int magnitude = stoi(input.substr(pos + 1));
        if (direction == "down") {
            aim += magnitude;
        } else if (direction == "up") {
            aim -= magnitude;
        } else if (direction == "forward") {
            hor += magnitude;
            depth += (magnitude*aim);
        }
    }
    cout<<depth*hor<<endl;
}
