# AtCoder Beginner Contest 237

## Problem A: Not Overflow

### Solution: read input

```c++
long long N;
cin>>N;
if (N<=INT32_MAX && N>=INT32_MIN) {
    cout<<"Yes"<<endl;
} else {
    cout<<"No"<<endl;
}
```

## Problem B: Matrix Transposition

### Solution: Iterate and create new array

Could be improved by doing with O(1) space

```c++
int H, W, a;
cin >> H >> W;
vector<vector<int>> A(H, vector<int>(W,0)), B(W, vector<int>(H,0));
for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
        cin >> a;
        A[i][j] = a;
    }
}

for (int i = 0;i<W;i++) {
    for (int j = 0;j<H;j++) {
        B[i][j] = A[j][i];
    }
}
for (int i = 0;i<W;i++) {
    for (int j = 0;j<H;j++) {
        cout << B[i][j] << " ";
    }
    cout<<endl;
}
```

## Problem C: kasaka

### Solution: Iterate through all trailing a's and then check if it is palindrome


```c++
bool isPalindrome(string& S, int i, int j) {
    while (i<j) {
        if (S[i++]!=S[j--]) return false;
    }
    return true;
}

int main() {
    string S;
    cin>>S;
    int N = S.size(), i = 0, j = N-1;
    while (i<j) {
        if (S[i]=='a' && S[j]=='a') {
            i++;
            j--;
        } else if (S[j]=='a') {
            j--;
        } else {
            break;
        }
    }
    if (isPalindrome(S, i, j)) {
        cout << "Yes" << endl;
    } else {
        cout << "No" << endl;
    }
}
```

## Problem D: LR Insertion

### Solution:  Construct a binary tree + inorder traversal

```c++
struct Node {
    int val;
    Node *left, *right;
    Node(int val) : val(val), left(nullptr), right(nullptr) {}
};

void buildTree(Node* root, string& S) {
    for (int i = 0;i<S.size();i++) {
        if (S[i] == 'L') {
            root->left = new Node(i+1);
            root = root->left;
        } else {
            root->right = new Node(i+1);
            root = root->right;
        }
    }
}

void inorder(Node* root) {
    if (!root) return;
    inorder(root->left);
    cout << root->val << " ";
    inorder(root->right);
}

int main() {
    int N;
    string S;
    cin>>N>>S;
    Node *root = new Node(0);
    buildTree(root, S);
    inorder(root);
}
```

## Problem E: Skiing

### Solution: BFS + undirected weighted graph

```c++
int N, M, h, u, v,d;
cin>>N>>M;
vector<vector<int>> graph(N);
vector<long long> H, dist(N,INT_MIN);
dist[0] = 0;
for (int i = 0;i<N;i++) {
    cin>>h;
    H.push_back(h);
}
for(int i=0;i<M;i++){
    cin>>u>>v;
    u--;
    v--;
    graph[u].push_back(v);
    graph[v].push_back(u);
}
queue<pair<long long,int>> q;
q.emplace(0,0);
long long best = 0;
while (!q.empty()) {
    tie(d,u) = q.front();
    q.pop();
    if (d < dist[u]) continue;
    for (int nei : graph[u]) {
        long long diff = H[u]-H[nei];
        if (diff<0) {
            diff*=2;
        }
        long long ncost = d + diff;
        if (ncost>dist[nei]) {
            best = max(best, ncost);
            dist[nei] = ncost;
            q.emplace(ncost,nei);
        }
    }
}
cout<<best<<endl;
```

