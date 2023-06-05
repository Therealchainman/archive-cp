class Solution:
		def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
			s = ""
			mod = 10**9 + 7

			@lru_cache(None)
			def dfs(idx, tight, sm):  
				nonlocal s,min_sum,max_sum,mod

				if idx == len(s):
					if sm >= min_sum and sm <= max_sum:
						return 1
					return 0

				up = int(s[idx]) if tight else 9  
				res = 0
				for digit in range(up + 1):
					newSum = sm + digit
					if newSum > max_sum: # next digits are more greater than curr so newSum always greater
						break
					res += dfs(idx + 1, tight and digit == up, newSum)
					res %= mod
				return res


			s = num2
			res = dfs(0,1,0)

			dfs.cache_clear()  # clear the dp states for new dfs

			s = str(int(num1)-1)
			res -= dfs(0,1,0)

			return res % mod