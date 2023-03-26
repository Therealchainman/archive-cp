# Leetcode Weekly contest 270

## 2094. Finding 3-Digit Even Numbers

### Solution: simplify 3 for loops by using fact only 3 digits and count the digits. 

```c++
vector<int> findEvenNumbers(vector<int>& digits) {
    int count[10] = {};
    for (int& dig : digits) {
        count[dig]++;
    }
    vector<int> arr;
    for (int i = 1;i<10;i++) {
        for (int j = 0;j<10 && count[i]>0;j++) {
            for (int k = 0;k<10 && count[j]>(i==j);k+=2) {
                if (count[k]>(j==k)+(i==k)) {
                    arr.push_back(i*100+j*10+k);
                }
            }
        }
    }
    return arr;
}
```

## 2095. Delete the Middle Node of a Linked List

### Solution: 

```c++
ListNode* deleteMiddle(ListNode* head) {
    if (!head->next) {
        return nullptr;
    }
    ListNode *slow = head, *fast = head, *prev = head;
    while (fast && fast->next) {
        prev = slow;
        slow=slow->next;
        fast=fast->next->next;
    }
    prev->next=slow->next;
    return head;
}
```

## 2096. Step-By-Step Directions From a Binary Tree Node to Another

### Solution: Find the LCA with DFS, reconstruct the paths with O(n) space

```c++
vector<vector<TreeNode*>> paths;
int start, end;
void dfs(TreeNode* root, vector<TreeNode*>& path) {
    if (!root) {
        return;
    }
    path.push_back(root);
    if (root->val==start) {
        paths[0]=path;
    }
    if (root->val==end) {
        paths[1]=path;
    }
    dfs(root->left, path);
    dfs(root->right, path);
    path.pop_back();
}
string getDirections(TreeNode* root, int startValue, int destValue) {
    start = startValue, end = destValue;
    vector<TreeNode*> path;
    paths.resize(2);
    dfs(root, path);
    TreeNode* lca;
    int ups = 0;
    for (int i = 0;;i++) {
        if (i==paths[1].size()) {
            ups = paths[0].size()-i;
            lca = paths[1].back();
            break;
        }
        if (i==paths[0].size()) {
            lca = paths[0].back();
            break;
        }
        if (paths[0][i]!=paths[1][i]) {
            ups = paths[0].size()-i;
            lca = paths[1][i-1];
            break;
        }
    }
    string directions;
    while (ups--) {
        directions += 'U';
    }
    for (int i = 0;i<paths[1].size();i++) {
        if (lca->left==paths[1][i]) {
            directions += 'L';
            lca = lca->left;
        } else if (lca->right==paths[1][i]) {
            directions += 'R';
            lca=lca->right;
        }
    }
    return directions;
}
```

### Solution: LCA with O(1) space? can that be used here? 



## 2097. Valid Arrangement of Pairs

### Solution: Eulerian path/circuit with Hierzholzer's algorithm, indegrees,outdegrees + postorder dfs and reverse traversed edges.
Eulerian path is when you visit each edge exactly once.

```c++
vector<vector<int>> path;
unordered_map<int,int> indegrees,outdegrees;
unordered_map<int,vector<int>> graph;
void dfs(int node) {
    while (outdegrees[node]) {
        outdegrees[node]--;
        int nei = graph[node][outdegrees[node]];
        dfs(nei);
        path.push_back({node,nei});
    }
}
vector<vector<int>> validArrangement(vector<vector<int>>& pairs) {
    unordered_set<int> nodes;
    for (auto& pir : pairs) {
        indegrees[pir[1]]++;
        outdegrees[pir[0]]++;
        nodes.insert(pir[0]);
        nodes.insert(pir[1]);
        graph[pir[0]].push_back(pir[1]); // directed edge form 0 -> 1 
    }
    int start = pairs[0][0];
    for (auto& node:nodes) {
        if (outdegrees[node]-indegrees[node]==1) {
            start = node;
            break;
        }
    }
    dfs(start);
    reverse(path.begin(),path.end());
    return path;
}
```