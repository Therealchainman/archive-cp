# Codeforces Round 943 Div 3

## F. Equal XOR Segments

### Solution 1:  bit manipulation, prefix xor, range parity queries

```py
from collections import defaultdict
import bisect
def main():
    n, q = map(int, input().split())
    arr = list(map(int, input().split()))
    groups = defaultdict(list)
    prefix = [0] * n 
    for i in range(n):
        prefix[i] = arr[i]
        if i > 0: prefix[i] ^= prefix[i - 1]
        groups[prefix[i]].append(i)
    ans = [0] * q
    for i in range(q):
        l, r = map(int, input().split())
        l -= 1; r -= 1
        target = prefix[r] ^ (prefix[l - 1] if l > 0 else 0)
        bmask = prefix[l - 1] if l > 0 else 0
        if target == 0:
            j = bisect.bisect_left(groups[bmask], l)
            if j < len(groups[bmask]) and groups[bmask][j] <= r: ans[i] = 1
        else:
            mask = target ^ bmask
            j = bisect.bisect_left(groups[mask], l)
            if j == len(groups[mask]) or groups[mask][j] >= r: continue
            l = groups[mask][j]
            bmask = prefix[l]
            mask = target ^ bmask
            j = bisect.bisect_left(groups[mask], l)
            if j == len(groups[mask]) or groups[mask][j] >= r: continue
            l = groups[mask][j]
            bmask = prefix[l]
            if prefix[r] ^ bmask == target: ans[i] = 1
    for x in ans:
        print("YES" if x else "NO")
    print()

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G1. Division + LCP (Easy version)

### Solution 1:  binary search, z algorithm

```py
def z_algorithm(s: str) -> list[int]:
    n = len(s)
    z = [0]*n
    left = right = 0
    for i in range(1,n):
        # BEYOND CURRENT MATCHED SEGMENT, TRY TO MATCH WITH PREFIX
        if i > right:
            left = right = i
            while right < n and s[right-left] == s[right]:
                right += 1
            z[i] = right - left
            right -= 1
        else:
            k = i - left
            # IF PREVIOUS MATCHED SEGMENT IS NOT TOUCHING BOUNDARIES OF CURRENT MATCHED SEGMENT
            if z[k] < right - i + 1:
                z[i] = z[k]
            # IF PREVIOUS MATCHED SEGMENT TOUCHES OR PASSES THE RIGHT BOUNDARY OF CURRENT MATCHED SEGMENT
            else:
                left = i
                while right < n and s[right-left] == s[right]:
                    right += 1
                z[i] = right - left
                right -= 1
    return z

def main():
    n, l, r = map(int,input().split())
    s = input()
    ans = [0] * (n + 1)
    zarr = z_algorithm(s)
    for i in range(n):
        zarr[i] = min(i, zarr[i])
    ans[1] = n
    memo = [-1] * (n + 1)
    def seg_count(target):
        if memo[target] != -1: return memo[target]
        count = 1
        prv = -1
        for i in range(n):
            if zarr[i] >= target and i - target >= prv: 
                count += 1
                prv = i
        memo[target] = count
        return count
    for seg in range(max(2, l), r + 1):
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if seg_count(mid) >= seg:
                lo = mid
            else:
                hi = mid - 1
        ans[seg] = lo
        if lo == 0: break
    print(*ans[l : r + 1])

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G2. Division + LCP (hard version)

### Solution 1:  memoization with binary search, z algorithm

```cpp
std::vector<int> z_algorithm(const std::string& s) {
    int n = s.length();
    std::vector<int> z(n, 0);
    int left = 0, right = 0;
    for (int i = 1; i < n; ++i) {
        if (i > right) {
            left = right = i;
            while (right < n && s[right-left] == s[right]) {
                right++;
            }
            z[i] = right - left;
            right--;
        } else {
            int k = i - left;
            if (z[k] < right - i + 1) {
                z[i] = z[k];
            } else {
                left = i;
                while (right < n && s[right-left] == s[right]) {
                    right++;
                }
                z[i] = right - left;
                right--;
            }
        }
    }
    return z;
}

int main() {
    int T;
    std::cin >> T;
    while (T--) {
        int n, l, r;
        std::cin >> n >> l >> r;
        std::string s;
        std::cin.ignore(); // to skip the newline character left in the input buffer
        std::getline(std::cin, s);
        std::vector<int> ans(n + 1, 0);
        std::vector<int> zarr = z_algorithm(s);
        for (int i = 0; i < n; ++i) {
            zarr[i] = std::min(i, zarr[i]);
        }
        ans[1] = n;
        std::vector<int> memo(n + 1, -1);
        auto seg_count = [&](int target) -> int {
            if (memo[target] != -1) return memo[target];
            int count = 1;
            int prv = -1;
            for (int i = 0; i < n; ++i) {
                if (zarr[i] >= target && i - target >= prv) {
                    count++;
                    prv = i;
                }
            }
            memo[target] = count;
            return count;
        };
        for (int seg = std::max(2, l); seg <= r; ++seg) {
            int lo = 0, hi = n;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (seg_count(mid) >= seg) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            ans[seg] = lo;
            if (lo == 0) break;
        }
        for (int i = l; i <= r; ++i) {
            std::cout << ans[i] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
```