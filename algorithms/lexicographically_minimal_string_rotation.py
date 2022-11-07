from collections import deque
"""
tournament algorithm with dual elimination
Find the champion in the tournament of 1v1 contests
"""
def min_string_rotation(s: str) -> str:
    min_char = min(s)
    s_len = len(s)
    advance = lambda index: (index + 1)%s_len
    champions = deque()
    for i, ch in enumerate(s):
        if ch == min_char:
            champions.append(i)
    # DUAL ELIMINATION UNTIL ONE CHAMPION REMAINS
    while len(champions) > 1:
        champion1 = champions.popleft()
        champion2 = champions.popleft()
        # ASSUME CHAMPION1 IS SMALLER INDEX
        if champion2 < champion1:
            champion1, champion2 = champion2, champion1
        # length of substring for champions is champion2-champion1
        # abcdefg
        # ^  ^
        # substring should be abc for champion 1, and def for champion 2
        current_champion = champion1
        left_champion, right_champion = champion1, champion2
        for _ in range(champion2 - champion1):
            if s[left_champion] < s[right_champion]: break
            if s[left_champion] > s[right_champion]:
                current_champion = champion2
                break
            left_champion = advance(left_champion)
            right_champion = advance(right_champion)
        champions.append(current_champion)
    champion_index = champions.pop()
    return s[champion_index:] + s[:champion_index]

"""
C++ implementation of the algorithm above for reference

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
"""