#include <bits/stdc++.h>
using namespace std;

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