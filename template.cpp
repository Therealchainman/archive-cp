#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <deque>
#include <queue>
#include <numeric>
#include <stack>
#include <cassert>
#include <cstring>
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <limits>
#include <functional>
#include <tuple>
#include <chrono>
using namespace std;
#define endl '\n'
#define LOCAL

using int64 = int64_t;
using uint64 = unsigned long long;
using int128 = __int128_t;

#ifdef LOCAL
#include "src/debug.h"
#else
#define debug(...) 42
#endif

void solve() {
    
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    auto t0 = std::chrono::steady_clock::now();
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    debug("Elapsed: ", ms, " ms", "\n");
    return 0;
}
