# Classes


```cpp
struct Item {
    int key, prior, size;
    Item *l, *r;
    Item() {};
    Item(int key) : key(key), prior(rand()), size(1), l(NULL), r(NULL) {};
    Item(int key, int prior) : key(key), prior(prior), size(1), l(NULL), r(NULL) {};

};
```

## Example of using static variables in C++

This is a way to avoid recomputing something, have it only initialize once when the class is initialized, and have it be accessible to all instances of the class.  This is where static variable comes into play.  You also need a boolean to avoid recomputing, otherwise it will recalculate everything regardless of fact it is static.  The static aspect just means it is shared across all instances of the class. 

```cpp
const int MAXN = 50'001;

class Solution {
private:
    static vector<int> divisors[MAXN];
    static bool precomputed;
    static void precompute() {
        if (precomputed) return;
        for (int x = 1; x < MAXN; x++) {
            for (int y = x; y < MAXN; y += x) {
                divisors[y].push_back(x);
            }
        }
        precomputed = true;
    }
public:
    vector<int> gcdValues(vector<int>& nums, vector<long long>& queries) {
        precompute();
        ...
    }
};
vector<int> Solution::divisors[MAXN];
bool Solution::precomputed = false;
```