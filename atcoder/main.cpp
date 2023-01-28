#include <bits/stdc++.h>
using namespace std;

int neutral = 0;

struct FenwickTree {
    vector<int> nodes;
    
    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, int val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int query(int left, int right) {
        return query(right) - query(left);
    }

    int query(int idx) {
        int result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }

};


vector<FenwickTree> trees(26, FenwickTree());
vector<int> freq(26, 0);

bool is_sorted(vector<int> &range_char_freq, int left, int right) {
    for (int i = 0; i<26; i++) {
        if (left == right) break;
        if (range_char_freq[i] == 0) continue;
        if (trees[i].query(left, left + range_char_freq[i]) != range_char_freq[i]) return false;
        left += range_char_freq[i];
    }
    return true;
}

bool check(int min_i, int max_i, vector<int> &range_char_freq) {
    for (int i = min_i + 1; i < max_i; i++) {
        if (range_char_freq[i] != freq[i]) return false;
    }
    return true;
}

int main() {
    int n, q, idx, left, right, query_type;
    string s;
    char ch;
    cin>>n>>s>>q;

    for (int i = 0; i < 26; i++) {
        trees[i].init(n);
    }

    for (int i = 0; i < n; i++) {
        trees[s[i] - 'a'].update(i + 1, 1);
        freq[s[i] - 'a']++;
    }
    
    while (q--) {
        cin>>query_type;
        if (query_type == 1) {
            cin>>idx>>ch;
            idx--;
            int old_char = s[idx] - 'a';
            int new_char = ch - 'a';
            trees[old_char].update(idx + 1, -1);
            trees[new_char].update(idx + 1, 1);
            s[idx] = ch;
            freq[old_char]--;
            freq[new_char]++;
        } else {
            cin>>left>>right;
            left--;
            bool ans = true;
            int min_i = 0, max_i = 25;
            vector<int> range_char_freq(26, 0);
            for (int i = 0; i < 26; i++) {
                if (freq[i] == 0) continue;
                int curr = trees[i].query(left, right);
                range_char_freq[i] = curr;
            }
            while (range_char_freq[min_i] == 0) min_i++;
            while (range_char_freq[max_i] == 0) max_i--;
            if (!check(min_i, max_i, range_char_freq)) {
                cout << "No" << endl;
                continue;
            }

            if (is_sorted(range_char_freq, left, right)) {
                cout << "Yes" << endl;
            } else {
                cout << "No" << endl;
            }
        }
    }
    return 0;
}