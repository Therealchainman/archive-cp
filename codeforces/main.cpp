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

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

const int N = 1e5 + 5;

vector<long long> values;
vector<vector<int>> adj_list;
unordered_map<long long, int> min_ops[N];

int is_leaf(int node, int parent) {
    return adj_list[node].size() == 1 && parent != -1;
}

int dfs(int node, int parent) {
    if (is_leaf(node, parent)) {
        min_ops[node][values[node]] = 0;
        return values[node];
    }
    unordered_map<long long, int> freq;
    int operation_count = 0;
    for (int child : adj_list[node]) {
        if (child == parent) continue;
        long long val = dfs(child, node);
        int ops = min_ops[child][val];
        freq[val]++;
        operation_count += ops;
    }
    // int maxer = max_element(freq.begin(), freq.end(), [](const pair<int, int>& x, const pair<int, int>& y){ return x.second < y.second; })->second;
    // int sum = accumulate(freq.begin(), freq.end(), 0, [](const int& a, const auto& b) { return a + b.second; });
    // printf("max_count: %d, sum: %d\n", maxer, sum);
    int total_cost = operation_count + accumulate(freq.begin(), freq.end(), 0, [](const int& a, const auto& b) { return a + b.second; }) - max_element(freq.begin(), freq.end(), [](const pair<int, int>& x, const pair<int, int>& y){ return x.second < y.second; })->second;
    long long max_value = max_element(freq.begin(), freq.end(), [](const pair<int, int>& x, const pair<int, int>& y){ return x.second < y.second; })->first;
    long long new_val = max_value ^ values[node];
    min_ops[node][new_val] = total_cost;
    // printf("node: %d, new_val: %lld, max_val: %lld, total_cost: %d\n", node, new_val, max_value, total_cost);
    min_ops[node][0] = total_cost + (new_val != 0);
    return new_val;
}

int main() {
    int n = read();
    values.resize(n + 1);
    adj_list.resize(n + 1);
    for (int i = 1; i <= n; i++) {
        // long long val = readll();
        values[i] = readll();
    }
    for (int i = 1; i < n; i++) {
        int u = read(), v = read();
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }
    dfs(1, -1);
    cout << min_ops[1][0] << endl;
    return 0;
}