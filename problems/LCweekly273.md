# Leetcode Weekly Contest 273

## 2119. A Number After a Double Reversal

### Solution 1: check that number doesn't have trailing zeros via it is not divisible by 10

```c++
bool isSameAfterReversals(int num) {
    return num==0 || num%10!=0;
}
```

## 2120. Execution of All Suffix Instructions Staying in a Grid

### Solution 1: Brute force algorithm to execute all instructions until out of bounds

```c++
vector<int> executeInstructions(int n, vector<int>& startPos, string s) {
    int m = s.size();
    vector<int> results(m,0);
    auto inBounds = [&](const auto i, const auto j) {
        return i>=0 && i<n && j>=0 && j<n;  
    };
    for (int i = 0;i<m;i++) {
        int steps = 0;
        for (int j = i, row = startPos[0], col = startPos[1];j<m;j++) {
            col += (s[j]=='R');
            col -= (s[j]=='L');
            row += (s[j]=='D');
            row -= (s[j]=='U');
            if (!inBounds(row,col)) {
                break;
            }
            steps++;
        }
        results[i] = steps;
    }
    return results;
}
```

### Solution 2:  O(m) with backtracking?

```c++

```


## 2121. Intervals Between Identical Elements

### Solution 1: Math, prefix ans suffix 

```c++
vector<long long> getDistances(vector<int>& arr) {
    unordered_map<int,vector<long long>> indicesMap;
    vector<long long> results((int)arr.size(),0);
    for (int i = 0;i<arr.size();i++) {
        indicesMap[arr[i]].push_back(i);
    }
    for (auto &[_, idx]: indicesMap) {
        long long prefix = accumulate(idx.begin(),idx.end(),0LL), n = idx.size();
        long long suffix = n*idx[0];
        results[idx[0]] = total - negate;
        for (int i = 1,j=n-1;i<n;i++,j--) {
            prefix += (idx[i]-idx[i-1])*i;
            suffix -= (j*idx[i-1]-j*idx[i]);
            results[idx[i]] = prefix - suffix;
        }
    }
    return results;
}
```

## LC 2122. Recover the Original Array

### Solution 1: brute force through k with hash map for frequency and build result

```c++
vector<int> recoverArray(vector<int>& nums) {
    unordered_map<int,int> freqs;
    for (int &num : nums) {
        freqs[num]++;
    }
    int n = nums.size();
    sort(nums.begin(),nums.end());
    int start = nums[0];
    for (int k = 1; ; ++k) {
        auto it = lower_bound(nums.begin(),nums.end(),start+2*k);
        int diff = *it-start;
        if (diff%2==1) {
            k = diff/2;
            continue;
        }
        k = diff/ 2;
        vector<int> res;
        auto ff = freqs;
        for (int &num: nums) {
            if (ff[num]==0) continue;
            ff[num]--;
            if (ff[num+diff]--==0) {
                break;
            }
            res.push_back(num+k);
        }
        if (res.size()==n/2)
            return res;
    }
    return {};add and remove 
}
```
### Solution 2: Instead of looping through k, we loop through all all elements and pair up with the 0th 


```c++
vector<int> recoverArray(vector<int>& nums) {
    unordered_map<int,int> freqs;
    for (int &num : nums) {
        freqs[num]++;
    }
    int n = nums.size();
    sort(nums.begin(),nums.end());
    int start = nums[0];
    for (int i = 1, k;i<n;i++) {
        int diff = nums[i]-nums[0];
        if (diff==0 || diff%2==1) {
            continue;
        }
        k = diff/ 2;
        vector<int> res;
        auto ff = freqs;
        for (int &num: nums) {
            if (ff[num]==0) continue;
            ff[num]--;
            if (ff[num+diff]--==0) {
                break;
            }
            res.push_back(num+k);
        }
        if (res.size()==n/2)
            return res;
    }
    return {};
}
```


### Solution 3: Using multiset with similar idea as above

```c++
vector<int> recoverArray(vector<int>& nums) {
    multiset<int> s(nums.begin(),nums.end());
    int start = *s.begin();
    for (int k = 1;;k++) {
        auto it = s.lower_bound(start+2*k);
        k = (*it-start)/2;
        if (start+2*k!=*it) continue;
        vector<int> recovered;
        auto ss = s;
        while (!ss.empty()) {
            auto it_lower = ss.begin();
            auto it_higher = ss.find(*it_lower+2*k);
            if (it_higher==ss.end()) break;
            recovered.push_back(*it_lower+k);
            ss.erase(it_lower);
            ss.erase(it_higher);
        }
        if (ss.empty()) {
            return recovered;
        }
    }
    return {};
}
```