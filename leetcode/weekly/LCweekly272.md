# Leetcode Weekly contest 272

## LC2108. Find First Palindromic String in the Array

### Solution:  Reverse string

Check if it is palindrome by reversing each word and return the first one that is palindrome


```c++
bool isPalindrome(string& word) {
    string rword = "";
    int n = word.size();
    for (int i = n-1;i>=0;i--) {
        rword += word[i];
    }
    return word==rword;
}
string firstPalindrome(vector<string>& words) {
    for (string &word : words) {
        if (isPalindrome(word)) {
            return word;
        }
    }
    return "";
}
```

```c++
string firstPalindrome(vector<string>& words) {
    for (string &word : words) {
        if (word==string(word.rbegin(),word.rend())) {
            return word;
        }
    }
    return "";
}
```

```py
def firstPalindrome(self, words: List[str]) -> str:
    for w in words:
        if w==w[::-1]:
            return w   
    return ""
```

### Solution: Two pointers to find palindromes

Efficient solution, avoid creating temporary strings.  

```c++
bool isPalindrome(string& s) {
    int n = s.size();
    for (int i = 0,j = n-1;i<j;i++,j--) {
        if (s[i]!=s[j]) {return false;}
    }
    return true;
}
string firstPalindrome(vector<string>& words) {
    for (string &word : words) {
        if (isPalindrome(word)) {
            return word;
        }
    }
    return "";
}
```

## LC2109. Adding Spaces to a String

### Solution: String 

I am able to use the reserve to set the capacity to reduce the time for reallocating the string when it reaches capacity. 
This can be done because I know the final size of the string. 

```c++
string addSpaces(string s, vector<int>& spaces) {
    string ret;
    int n = s.size(), m = spaces.size();
    ret.reserve(n+m);
    for (int i = 0,j=0;i<n;i++) {
        if (j<spaces.size() && spaces[j]==i) {
            ret += " ";
            j++;
        }
        ret += s[i];
    }
    return ret;
}
```

In python it is a little trickier with strings because it recreates a string when you add to it,
so you have to use an array instead and append to it, then reconstruct the string from the array. 

```py
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        store = []
        store.append(s[:spaces[0]])
        for i in range(1,len(spaces)):
            store.append(' ')
            store.append(s[spaces[i-1]:spaces[i]])
        store.append(' ')
        store.append(s[spaces[-1]:])
        return "".join(store)
```

## LC2110. Number of Smooth Descent Periods of a Stock


### Solution: Iterative

```c++
long long getDescentPeriods(vector<int>& prices) {
    long long num = 1, len = 1;
    int n= prices.size();
    for (int i = 1;i<n;i++) {
        len = prices[i-1]-prices[i]==1 ? len+1 : 1;
        num+=len;
    }
    return num;
}
```

## LC2111. Minimum Operations to Make the Array K-Increasing

This problem you can use the idea of longest non-decreasing subsequence to solve it. 

This requires the O(nlogn) solution to that type of problem with the patience algorithm

I split the array up into k arrays that I will then compute the length of the longest non-decreasing subsequence in each array and subtract
it from the length of each array. 

take 
[12,6,12,6,14,2,13,17,3,8,11,7,4,11,18,8,8,3] with k=1
We find the length of the longest non-decreasing subsequence such as 2,3,8,11,11,18, which is of size 6
Now all we have to do is change the rest of the values to make the entire subarray non-decreasing, so yeah we only need to 
change 12.  

I guess since you are finding the minimum operations to make the array non-decreasing, you find the longest non-decreasing subsequence
and just change the rest of the values.  



### Solution: binary search with vector to find the k longest non-decreasing subsequences

```c++
const int NEUTRAL = 1e9;
int patience(vector<int>& arr) {
    int n = arr.size(), len = 0;
    vector<int> T(n,NEUTRAL);
    for (int &num : arr) {
        int i = upper_bound(T.begin(),T.end(), num) - T.begin();
        len = max(len, i+1);
        T[i] = num;
    }
    return len;
}
int kIncreasing(vector<int>& arr, int k) {
    vector<int> karray;
    int n = arr.size(), cnt = 0;
    for (int i = 0;i<k;i++) {
        karray.clear();
        for (int j = i;j<n;j+=k) {
            karray.push_back(arr[j]);
        }
        cnt += (karray.size()-patience(karray));
    }
    return cnt;
}
```

### Solution: This one is same but using a monostack 


```c++
int kIncreasing(vector<int>& arr, int k) {
    vector<int> karray;
    int n = arr.size(), longest = 0;
    for (int i = 0;i<k;i++) {
        vector<int> mono;
        for (int j = i;j<n;j+=k) {
            if (mono.empty() || mono.back()<=arr[j]) {
                mono.push_back(arr[j]);
            } else {
                *upper_bound(mono.begin(),mono.end(),arr[j]) = arr[j];
            }
        }
        longest += mono.size();
    }
    return arr.size()-longest;
}
```


I need practice using bisect_right in python, this is equivalent to upper_bound that is it finds the index of the first element
that is strictly greater than what you are searching for.  

```py
def kIncreasing(self, arr: List[int], k: int) -> int:
    def LNDS(arr):
        mono = []
        for x in arr:
            if not mono or mono[-1]<=x:
                mono.append(x)
            else:
                mono[bisect_right(mono, x)] = x
        return len(mono)
    return len(arr) - sum(LNDS(arr[i::k]) for i in range(k))
```