#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, K, p;
    cin>>N>>K;
    priority_queue<int, vector<int>, greater<int>> minHeap;
    vector<int> results;
    for (int i = 0;i<N;i++) {
        cin>>p;
        minHeap.push(p);
        if (minHeap.size()>K) {
            minHeap.pop();
        }
        if (minHeap.size()==K) {
            results.push_back(minHeap.top());
        }
    }
    for (int &res : results) {
        cout<<res<<endl;
    }
}