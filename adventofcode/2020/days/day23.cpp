#include "../libraries/aoc.h"

struct Node {
    Node* next;
    ll val;
};

struct Cups {
    Node* cups;
    vector<Node*> indexVec;

    explicit Cups() {
        vector<Node*> indices(1000001, new Node);
        indexVec = indices;
    }
};

struct Game {
    Cups c;
    ll destVal;
    Node* rmNode;
    Node* destNode;
    set<ll> picked;
};

Game g;

void playGame(bool part2) {
    Node* next;
    Node* head;
    head = g.c.cups;
    if (part2) {
        for (int i = 0;i<8;i++) {
            g.c.cups=g.c.cups->next;
        }
        for (int i = 10;i<=1000000;i++) {
            next = new Node;
            next->val = i;
            g.c.cups->next=next;
            g.c.cups = g.c.cups->next;
            g.c.indexVec[next->val] = next;
        }
        g.c.cups->next = head;
    }
    ll moves = part2 ? 10000000 : 100;
    Node* curCup = head;
    Node* curNode;
    Node* lastNode;
    Node* afterDestNode;
    for (int j = 0;j<moves;j++) {
        // Removing my three nodes
        curNode = curCup->next;
        g.rmNode = curNode;
        for (int i = 0;i<3;i++) {
            g.picked.insert(curNode->val);
            lastNode = curNode;
            curNode = curNode->next;
        }
        if (part2) {
            g.destVal = curCup->val-1==0 ? 1000000 : curCup->val-1;
        } else {
            g.destVal = curCup->val-1==0 ? 9: curCup->val-1;
        }
        curCup->next = curNode;
        lastNode->next = nullptr;
        // Searching for the destination value
        while (g.picked.count(g.destVal)>0) {
            g.destVal--;
            if (g.destVal==0) {
                g.destVal= part2 ? 1000000 : 9;
            }
        }
        // Insert the three nodes after the destination node. 
        g.destNode = g.c.indexVec[g.destVal];
        afterDestNode = g.destNode->next;
        g.destNode->next = g.rmNode;
        while (g.rmNode->next!=nullptr) {
            g.rmNode = g.rmNode->next;
        }
        g.rmNode->next = afterDestNode;
        g.picked.clear();
        curCup = curCup->next;
        Node* test = curCup;
    }
    if (part2) {
        ll res = 1;
        g.destNode = g.c.indexVec[1];
        for (int i = 0;i<2;i++) {
            g.destNode = g.destNode->next;
            cout<<g.destNode->val<<endl;
            res*=g.destNode->val;
        }
        cout<<res<<endl;
    } else {
        string res = "";
        g.destNode = g.c.indexVec[1];
        for (int i = 0;i<8;i++) {
            g.destNode = g.destNode->next;
            res+=to_string(g.destNode->val);
        }
        cout<<res<<endl;
    }
}

int main() {
    auto start = high_resolution_clock::now();
    freopen("inputs/big.txt","r",stdin);
    string input;
    Cups c = Cups();
    getline(cin,input);
    Node* node = new Node;
    Node* next;
    Node* head = node;
    c.cups = node;
    int i = 0;
    for (char ch : input) {
        i++;
        next = new Node;
        node->val=ch-'0';
        // Want the next to be nullptr for the last one. 
        node->next=next;
        c.indexVec[node->val]=node;
        if (i==input.size()) {
            continue;
        }
        node= node->next;
    }
    node->next=head;
    c.cups=head;
    g.c=c;
    // part 1
    cout<<"Part 1: ";
    playGame(false);
    // part 2
    cout<<"Part 2: ";
    playGame(true);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop-start);
    cout<<"Time: "<<duration.count()<<endl;
    return 0;
}