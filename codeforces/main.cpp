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

int neutral = INT_MAX;

struct SegmentTree {
    vector<int> nodes;
    vector<int> next;
    int size;

    void init(int n, vector<int> &init_arr) {
        size = 1;
        while (size < n) size *= 2;
        nodes.assign(2 * size, neutral);
        next.assign(n, neutral);
        build(init_arr);
    }

    void build(vector<int> &init_arr) {
        unordered_map<int, int> last;
        for (int i = 0; i < init_arr.size(); i++) {
            int segment_idx = i + size -1;
            int val = init_arr[i];
            if (last.find(val) != last.end()) {
                nodes[segment_idx] = i - last[val];
                next[last[val]] = i;
            }
            last[val] = i;
            ascend(segment_idx);
        }
    }

    int calc_op(int x, int y) {
        return min(x, y);
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            segment_idx--;
            segment_idx >>= 1;
            int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
            nodes[segment_idx] = calc_op(nodes[left_segment_idx], nodes[right_segment_idx]);
        }
    }

    void update(int segment_idx, int val) {
        segment_idx += size - 1;
        nodes[segment_idx] = val;
        ascend(segment_idx);
    }

    int query(int left, int right) {
        vector<tuple<int, int, int>> stack;
        stack.push_back({0, size, 0});
        int result = neutral;
        while (!stack.empty()) {
            // BOUNDS FOR CURRENT INTERVAL AND IDX FOR TREE
            int segment_left_bound = get<0>(stack.back());
            int segment_right_bound = get<1>(stack.back());
            int segment_idx = get<2>(stack.back());
            stack.pop_back();
            // NO OVERLAP
            if (segment_left_bound >= right || segment_right_bound <= left) continue;
            // COMPLETE OVERLAP
            if (segment_left_bound >= left && segment_right_bound <= right) {
                result = calc_op(result, nodes[segment_idx]);
                continue;
            }
            // PARTIAL OVERLAP
            int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
            int mid = (segment_left_bound + segment_right_bound) >> 1;
            stack.push_back({mid, segment_right_bound, right_segment_idx});
            stack.push_back({segment_left_bound, mid, left_segment_idx});
        }
        return result;
    }
};

int main() {
    int n = read(), m = read();
    vector<int> arr(n);
    for (int i = 0; i < n; i++) {
        arr[i] = read();
    }
    vector<tuple<int, int, int>> queries;
    for (int i = 0; i < m; i++) {
        int left = read(), right = read();
        queries.push_back({left-1, right-1, i});
    }
    sort(queries.begin(), queries.end());
    SegmentTree minSegTree;
    minSegTree.init(n, arr);
    vector<int> answer;
    answer.assign(m, -1);
    int index = 0;
    for (int i = 0; i < m; i++) {
        int left = get<0>(queries[i]);
        int right = get<1>(queries[i]);
        int id = get<2>(queries[i]);
        while (index < left) {
            if (minSegTree.next[index] != neutral) minSegTree.update(minSegTree.next[index], neutral);
            index++;
        }
        int query_res = minSegTree.query(left, right+1);
        if (query_res != neutral) answer[id] = query_res;
    }
    for (int ans : answer) {
        cout << ans << endl;
    }
}