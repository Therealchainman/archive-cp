# Leetcode Weekly Contest 277

## 2148. Count Elements With Strictly Smaller and Greater Elements

### Solution: Compare all elements to min and max in array and check it is strictly between

TC: O(n)

```py
def countElements(self, nums: List[int]) -> int:
    mx, mn = max(nums), min(nums)
    return sum(1 for num in nums if num>mn and num<mx)
```

```c++
int countElements(vector<int>& nums) {
    int mn = INT_MAX, mx = INT_MIN;
    for (int& num : nums) {
        mn = min(mn, num);
        mx = max(mx,num);
    }
    return accumulate(nums.begin(),nums.end(),0,[&](const auto& a, const auto& b) {
        return a + (b>mn && b<mx);
    });
}
```

## 2149. Rearrange Array Elements by Sign

### Solution: Two pointers


```c++
vector<int> rearrangeArray(vector<int>& nums) {
    int n = nums.size();
    vector<int> pos
    for (int i = 0,pptr = 0, nptr = 1;i<n;i++) {
        if (nums[i]>0) {
            res[pptr] = nums[i];
            pptr+=2;
        } else {
            res[nptr] = nums[i];
            nptr+=2;
        }
    }
    return res;
}
```

## 2150. Find All Lonely Numbers in the Array

### Solution: hashmap 

```c++
vector<int> findLonely(vector<int>& nums) {
    unordered_map<int,int> count;
    for (int& num : nums) {
        count[num]++;
    }
    vector<int> res;
    for (int& num : nums) {
        if (count[num]==1 && count[num-1]+count[num+1]==0) {
            res.push_back(num);
        }
    }
    return res;
}
```

```py
def findLonely(self, nums: List[int]) -> List[int]:
    count = Counter(nums)
    return [num for num in nums if count[num]==1 and count[num-1]+count[num+1]==0]
```

## 2151. Maximum Good People Based on Statements

### Solution: bitmask + loop to check statements are valid

TC: O(n^2 * 2^n)

```c++
int maximumGood(vector<vector<int>>& statements) {
    int n = statements.size(), best = 0;
    auto valid = [&](int mask) {
        for (int i = 0;i<n;i++) {
            if ((mask>>i)&1) {
                for (int j = 0;j<n;j++) {
                    int good = (mask>>j)&1;
                    if ((statements[i][j]==0 && good) || (statements[i][j]==1 && !good)) return false;
                }
            }
        }  
        return true;
    };
    for (int i = 1; i<(1<<n);i++) {
        if (valid(i)) {
            best = max(best, __builtin_popcount(i));
        }
    }
    return best;
}
```

### Solution: dfs + backtracking (check if valid and update best answer with count of good)

```c++
int best;
bool valid(string& s, vector<vector<int>>& statements) {
    int n = statements.size();
    for (int i = 0;i<n;i++) {
        if (s[i]=='1') {
            for (int j = 0;j<n;j++) {
                int good = s[j]-'0';
                if ((statements[i][j]==0 && good) || (statements[i][j]==1 && !good)) return false;
            }
        }
    }  
    return true;
}
void dfs(vector<vector<int>>& statements, string& state, int i, int cnt) {
    if (i==statements.size()) {
        if (valid(state,statements)) {
            best = max(best, cnt);
        }
        return;
    }
    state += '0'; // assume bad person
    dfs(statements,state,i+1,cnt);
    state[i] = '1'; // assume good person
    dfs(statements,state,i+1,cnt+1);
    state.pop_back();
}
int maximumGood(vector<vector<int>>& statements) {
    string cur = "";
    best = 0;
    cur.reserve((int)statements.size());
    dfs(statements,cur,0,0);
    return best;
}
```