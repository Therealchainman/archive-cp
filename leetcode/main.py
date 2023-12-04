class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        dp = [0] * n
        freq = Counter()
        left = over_count = under_count = 0
        for right in range(n):
            ch = word[right]
            if right > 0 and abs(ord(word[right]) - ord(word[right - 1])) > k:
                freq.clear()
                left = right
                over_count = under_count = 0
            freq[ch] += 1
            if freq[ch] == 1 and freq[ch] < k:
                under_count += 1
            elif freq[ch] == k + 1:
                over_count += 1
            while over_count > 0:
                freq[word[left]] -= 1
                if freq[word[left]] == k:
                    over_count -= 1
                elif freq[word[left]] == 0 and k != 1:
                    under_count -= 1
                left += 1
            if under_count == 0:
                dp[right] = dp[left - 1] + 1 if left > 0 else 1
        return sum(dp)
    
"""
You have to know what is between two characters as well.  It is not just what is at the end. 


"""