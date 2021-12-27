# Leetcode Biweekly Contest 68

## 2114. Maximum Number of Words Found in Sentences

### Solution: get the maximum result by splitting all strings with the ' ' whitespace character into lists and get the length

```py
def mostWordsFound(self, sentences: List[str]) -> int:
    return max(len(sentence.split(' ')) for sentence in sentences)
```

```c++
int mostWordsFound(vector<string>& sentences) {
    int mx = 0;
    for (string& s : sentences) {
        stringstream ss(s);
        string tmp;
        int cnt = 0;
        while (getline(ss,tmp,' ')) {
            cnt++;
        }
        mx = max(mx,cnt);
    }
    return mx;
}
```

## 2115. Find All Possible Recipes from Given Supplies

### Solution: topological sort and bfs through the directed graph

```c++
vector<string> findAllRecipes(vector<string>& recipes, vector<vector<string>>& ingredients, vector<string>& supplies) {
    vector<string> results;
    int n = recipes.size();
    unordered_map<string, int> indegrees;
    unordered_map<string, vector<string>> graph;
    for (int i = 0;i<n;i++) {
        for (string& ingred : ingredients[i]) {
            graph[ingred].push_back(recipes[i]);
            indegrees[recipes[i]]++;
        }
    }
    queue<string> q;
    for (string &sup : supplies) {
        q.push(sup);
    }
    while (!q.empty()) {
        string ingredient = q.front();
        q.pop();
        for (auto& nei : graph[ingredient]) {
            if (--indegrees[nei]==0) {
                q.push(nei);
                results.push_back(nei);
            }
        }
    }
    return results;
}
```

## 2116. Check if parentheses String Can Be Valid


### Solution: 

```c++

```

"((()(()()))()((()()))))()((()(()"
"10111100100101001110100010001001"


## 2117. Abbreviating the Product of a Range

### Solution 1: Count the number of trailing zeroes for factorial(right)-factorial(left-1) and compute the prefix and suffix. Brute force if 
fewer than or equal to 10 digits.  

We don't want to count trailing zeroes in the fewer than 10 digits. 

The tricky part for me was computing the prefix.  I kind of just used a decimal or double in c++.  To compute the prefix
where I kept at least 5 digits before the decimal point.  Then compute it like that.  But to be frank it is a bit ignoreing precision and 
other errors potentially.  

```c++
const int MOD = 1e5;
class Solution {
public:
    int trailingZeroes(int n) {
        int cntFives = 0;
        for (int i = 5;i<=n;i*=5) {
            cntFives += (n/i);
        }
        return cntFives;
    }
    string abbreviateProduct(int left, int right) {
        long long prod = 1;
        int es = 0;
        for (long long i = left;i<=right;i++) {
            prod*=i;
            while (prod%10==0 && prod>0) {
                prod/=10;
                es++;
            }
            if (to_string(prod).size()>10) {
                break;
            }
        }
        if (to_string(prod).size()<=10) {
            return to_string(prod) + 'e' + to_string(es);
        }
        // solve the difficult case that has d>10, so it has prefix and suffix.  
        int zeroes = trailingZeroes(right)-trailingZeroes(left-1);
        int cntFives = zeroes, cntTwos = zeroes;
    
        // compute the suffix
        prod = 1;
        for (long long i = left;i<=right;i++) {
            prod*=i;
            while (prod%5==0 && cntFives>0) {
                prod/=5;
                cntFives--;
            }
            while (prod%2==0 && cntTwos>0) {
                prod/=2;
                cntTwos--;
            }
            prod%=MOD;
        }
        int leadingZeroes = 5-to_string(prod).size();

        string suffix = "";
        while (leadingZeroes--) {
            suffix += '0';
        }
        suffix += to_string(prod);
        // compute the prefix
        double pprod = 1.0;
        for (long long i = left;i<=right;i++) {
            pprod*=i;
            while (pprod>MOD) {
                pprod/=10.0;
            }
        }
        string prefix = to_string(int(pprod));
        return prefix + "..." + suffix + 'e'+to_string(zeroes);
    }
};
```