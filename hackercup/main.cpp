#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'

namespace fs = std::filesystem;

string name = "a.txt";

fs::path create_path(const std::string& directory, const std::string& file_name) {
    return fs::path(directory) / file_name;
}

string base = "substantial_losses";
string name = base + "_sample_input.txt";
// string name = base + "_validation_input.txt";
// string name = base + "_input.txt";

const int M = 998244353;
int W, G, L;

void solve() {
    cin >> W >> G >> L;
    int v = (2LL * L + 1) % M;
    int ans = ((W - G) * v) % M;
    cout << ans << endl;
}


signed main() {
    fs::path input_path = create_path("inputs", name);
    fs::path output_path = create_path("outputs", name);
    ifstream input_file(input_path);
    if (!input_file.is_open()) {
        std::cerr << "Error: Failed to open input file: " << input_path << endl;
        return 1;  // Exit with error if file cannot be opened
    }
    ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Failed to open output file: " << output_path << endl;
        return 1;  // Exit with error if file cannot be opened
    }
    cin.rdbuf(input_file.rdbuf());
    cout.rdbuf(output_file.rdbuf());
    solve();
    cin.rdbuf(nullptr);
    cout.rdbuf(nullptr);
    return 0;
}


/*
problem solve

You can output floats with using cout << fixed << setprecision(12) << p << endl;

This is to avoid stack overflow error

on linux:
run the following command to prevent stack overflow problems in recursive function calls. 
Needs to only be ran for each bash/shell 
ulimit -s unlimited
This is the compiler code that will add some optimizations with the -O2
g++ -O2 main.cpp -o main
If you want to debug then consider running this.
g++ -g main.cpp -o main
then you can run gdb ./main
and type run into the terminal and enter debug session.
And use `bt` to get the backtrace
on windows this works for compiling and preventing stack overflow problems.
g++ "-Wl,--stack=26843546" main.cpp -o main
*/
