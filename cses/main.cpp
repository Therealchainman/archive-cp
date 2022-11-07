#include <bits/stdc++.h>
using namespace std;

string min_string_rotation(string& s) {
    char min_char = *min_element(s.begin(), s.end());
    deque<int> champions;
    int n = s.size();
    for (int i = 0;i<n;i++) {
        if (s[i] == min_char) {
            champions.push_back(i);
        }
    }
    while (champions.size() > 1) {
        int champion1 = champions.front();
        champions.pop_front();
        int champion2 = champions.front();
        champions.pop_front();
        if (champion2 < champion1) swap(champion1, champion2);
        int current_champion = champion1;
        for (int left = champion1, right = champion2, sz = champion2-champion1; sz > 0; sz--, left++, right++) {
            if (left == n) left = 0;
            if (right == n) right = 0;
            if (s[left] < s[right]) break;
            if (s[left] > s[right]) {
                current_champion = champion2;
                break;
            }
        }
        champions.push_back(current_champion);
    }
    int champion_index = champions.front();
    return s.substr(champion_index) + s.substr(0, champion_index);
}

int main() {
    ios_base::sync_with_stdio(false);
	cin.tie(NULL);
    string s;
    cin>>s;
    cout<<min_string_rotation(s)<<endl;
}