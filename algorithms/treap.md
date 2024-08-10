# Treap

The data structure that is a combination of the Binary Search Tree and Heap

## Important

I first began creating treap using rand() in C++, but this is not actually that great for some reason. If you use the following instead it is much better, and just call rng()

```cpp
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
```

## Shandom Ruffle

Spliting the array into 5 subarrays and rearranging the subarrays into a new array.

```cpp
struct Item {
    int key, prior, size;
    Item *l, *r;
    Item() {};
    Item(int key) : key(key), prior(rand()), size(1), l(NULL), r(NULL) {};
    Item(int key, int prior) : key(key), prior(prior), size(1), l(NULL), r(NULL) {};

};
typedef Item* pitem;

// prints the in-order traversal of a tree
void output(Item *t) {
    if (!t) return;
    output(t -> l);
    cout << t -> key << " ";
    output(t -> r);
}

int size(const pitem &item) { return item ? item -> size : 0; }

void split(pitem t, pitem &l, pitem &r, int val) {
    if (!t) {
        l = r = NULL; 
        return;
    }
    if (size(t -> l) < val) {
        split(t->r, t->r, r, val - size(t -> l) - 1);
        l = t;
    }
    else {
        split(t->l, l, t->l, val);
        r = t;
    }
    t -> size = 1 + size(t -> l) + size(t -> r);
}

// merge left and right into t
void merge(pitem &t, pitem l, pitem r) {
    if (!l || !r) {
        t = l ? l : r;
        return;
    }
    if (l -> prior > r -> prior) {
        merge(l -> r, l -> r, r), t = l;
    }
    else {
        merge(r -> l, l, r -> l), t = r;
    }
    t -> size = 1 + size(t -> l) + size(t -> r);
}

int N;

void solve() {
    pitem root = nullptr;
    cin >> N;
    for (int i = 1; i <= N; i++) {
        merge(root, root, new Item(i));
    }
    for (int i = 0; i < N; i++) {
        int a, b;
        cin >> a >> b;
        a--; b--;
        if (a >= b) continue;
        pitem sp = nullptr, sa = nullptr, sm = nullptr, sb = nullptr, ss = nullptr;
        int length = min(N - b, b - a);
        split(root, sp, sa, a);
        split(sa, sa, sm, length);
        split(sm, sm, sb, b - a - length);
        split(sb, sb, ss, length);
        merge(root, sp, sb);
        merge(root, root, sm);
        merge(root, root, sa);
        merge(root, root, ss);
    }
    output(root);
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Pear TreaP

```cpp

```

## Sneetches and Speeches

```cpp
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

const int INF = 1e9;

struct Item {
    int val, prior, size, c0, p0, p1, s0, s1, mcons;
    int mir_prior, inv_prior, set0_prior, set1_prior;
    bool mirror, invert, set0, set1;
    Item *l, *r;
    Item() {};
    Item(int val) : val(val), prior(rng()), size(1), c0(val == 0 ? 1 : 0), p0(val == 0 ? 1 : 0), p1(val == 1 ? 1 : 0), s0(val == 0 ? 1 : 0), s1(val == 1 ? 1 : 0), mcons(1), mir_prior(INF), inv_prior(INF), set0_prior(INF), set1_prior(INF), mirror(false), invert(false), set0(false), set1(false), l(NULL), r(NULL) {};
};
typedef Item* pitem;

int size(const pitem &t) { return t ? t -> size : 0; }
int most_cons(const pitem &t) { return t ? t -> mcons : 0; }
int prefix0(const pitem &t) { return t ? t -> p0 : 0; }
int prefix1(const pitem &t) { return t ? t -> p1 : 0; }
int suffix0(const pitem &t) { return t ? t -> s0 : 0; }
int suffix1(const pitem &t) { return t ? t -> s1 : 0; }
int count0(const pitem &t) { return t ? t -> c0 : 0; }

void mirror(pitem &t) {
    t -> mirror = false;
    int p0 = prefix0(t), p1 = prefix1(t), s0 = suffix0(t), s1 = suffix1(t);
    t -> p0 = s0, t -> p1 = s1, t -> s0 = p0, t -> s1 = p1;
    swap(t -> l, t -> r);
    if (t -> l) {
        t -> l -> mirror ^= true;
        t -> l -> mir_prior = t -> mir_prior;
    }
    if (t -> r) {
        t -> r -> mirror ^= true;
        t -> r -> mir_prior = t -> mir_prior;
    }
}

void invert(pitem &t) {
    t -> invert = false;
    int p0 = prefix0(t), p1 = prefix1(t), s0 = suffix0(t), s1 = suffix1(t);
    t -> p0 = p1, t -> p1 = p0, t -> s0 = s1, t -> s1 = s0;
    t -> c0 = size(t) - count0(t);
    t -> val ^= 1;
    if (t -> l) {
        t -> l -> invert ^= true;
        t -> l -> inv_prior = t -> inv_prior;
    }
    if (t -> r) {
        t -> r -> invert ^= true;
        t -> r -> inv_prior = t -> inv_prior;
    }
}

void assign_zero(pitem &t) {
    t -> set0 = false;
    t -> p0 = size(t);
    t -> s0 = size(t);
    t -> p1 = 0;
    t -> s1 = 0;
    t -> c0 = size(t);
    t -> val = 0;
    if (t -> l) {
        t -> l -> set0 = true;
        t -> l -> set0_prior = t -> set0_prior;
    }
    if (t -> r) {
        t -> r -> set0 = true;
        t -> r -> set0_prior = t -> set0_prior;
    }
}

void assign_one(pitem &t) {
    t -> set1 = false;
    t -> p0 = 0;
    t -> s0 = 0;
    t -> p1 = size(t);
    t -> s1 = size(t);
    t -> c0 = 0;
    t -> val = 1;
    if (t -> l) {
        t -> l -> set1 = true;
        t -> l -> set1_prior = t -> set1_prior;
    }
    if (t -> r) {
        t -> r -> set1 = true;
        t -> r -> set1_prior = t -> set1_prior;
    }
}

void push(pitem t) {
    vector<pair<int, int>> ops;
    if (t) {
        if (t -> mirror) ops.push_back({t -> mir_prior, 1});
        if (t -> invert) ops.push_back({t -> inv_prior, 2});
        if (t -> set0) ops.push_back({t -> set0_prior, 3});
        if (t -> set1) ops.push_back({t -> set1_prior, 4});
    }
    sort(ops.begin(), ops.end());
    for (const auto &[_, id] : ops) {
        if (id == 1) {
            mirror(t);
        } else if (id == 2) {
            invert(t);
        } else if (id == 3) {
            assign_zero(t);
        } else if (id == 4) {
            assign_one(t);
        }
    }
}

void pull(pitem t) {
    if (t) {
        push(t -> l); push(t -> r);
        t -> size = 1 + size(t -> l) + size(t -> r);
        t -> c0 = count0(t -> l) + count0(t -> r) + (t -> val == 0);
        t -> p0 = prefix0(t -> l);
        t -> p1 = prefix1(t -> l);
        t -> s0 = suffix0(t -> r);
        t -> s1 = suffix1(t -> r);
        if (t -> val) {
            if (t -> p1 == size(t -> l)) t -> p1 += prefix1(t -> r) + 1;
            if (t -> s1 == size(t -> r)) t -> s1 += suffix1(t -> l) + 1;
        } else {
            if (t -> p0 == size(t -> l)) t -> p0 += prefix0(t -> r) + 1;
            if (t -> s0 == size(t -> r)) t -> s0 += suffix0(t -> l) + 1;
        }
        t -> mcons = max(most_cons(t -> l), most_cons(t -> r));
        if (t -> val) {
            t -> mcons = max(t -> mcons, suffix1(t -> l) + prefix1(t -> r) + 1);
        } else {
            t -> mcons = max(t -> mcons, suffix0(t -> l) + prefix0(t -> r) + 1);
        }
    }
}
  
void split(pitem t, pitem &l, pitem &r, int val) {
    push(t);
    if (!t) {
        l = r = NULL; 
    } else if (size(t -> l) < val) {
        split(t->r, t->r, r, val - size(t -> l) - 1);
        l = t;
    }
    else {
        split(t->l, l, t->l, val);
        r = t;
    }
    pull(t);
}
 
// merge left and right into t
void merge(pitem &t, pitem l, pitem r) {
    push(l); push(r);
    if (!l || !r) {
        t = l ? l : r;
    } else if (l -> prior > r -> prior) {
        merge(l -> r, l -> r, r);
        t = l;
    }
    else {
        merge(r -> l, l, r -> l);
        t = r;
    }
    pull(t);
}

void mirror(pitem t, int l, int r, int prior) {
    pitem t1, t2, t3;
    split(t, t1, t2, l);
    split(t2, t2, t3, r - l + 1);
    t2 -> mirror ^= true;
    t2 -> mir_prior = prior;
    merge(t, t1, t2);
    merge(t, t, t3);
}

void inversion(pitem t, int l, int r, int prior) {
    pitem t1, t2, t3;
    split(t, t1, t2, l);
    split(t2, t2, t3, r - l + 1);
    t2 -> invert ^= true;
    t2 -> inv_prior = prior;
    merge(t, t1, t2);
    merge(t, t, t3);
}

void range_sort(pitem t, int l, int r, int prior) {
    pitem pre, mid1, mid2, suf;
    split(t, pre, mid1, l); // split t into pre and mid1
    split(mid1, mid1, suf, r - l + 1); // split mid1 into mid1 and suf
    int c0 = count0(mid1);
    split(mid1, mid1, mid2, c0); // split mid1 into mid1 and mid2
    if (mid1) {
        mid1 -> set0 = true;
        mid1 -> set0_prior = prior;
    }
    if (mid2) {
        mid2 -> set1 = true;
        mid2 -> set1_prior = prior;
    }
    merge(t, pre, mid1);
    merge(t, t, mid2);
    merge(t, t, suf);
}
 
int N, M;
 
void solve() {
    pitem root = nullptr;
    cin >> N >> M;
    string s;
    cin >> s;
    for (char ch : s) {
        int v = ch - '0';
        merge(root, root, new Item(v));
    }
    for (int i = 0; i < M; i++) {
        int t, l, r;
        cin >> t >> l >> r;
        l--; r--;
        if (t == 1) {
            inversion(root, l, r, i);
            cout << most_cons(root) << endl;
        } else if (t == 2) {
            mirror(root, l, r, i);
            cout << most_cons(root) << endl;
        } else {
            range_sort(root, l, r, i);
            cout << most_cons(root) << endl;
        }
    }
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Cut and Paste

Split array, and rearrange

```cpp
struct Item {
    char key;
    int prior, size;
    Item *l, *r;
    Item() {};
    Item(char key) : key(key), prior(rand()), size(1), l(NULL), r(NULL) {};
    Item(char key, int prior) : key(key), prior(prior), size(1), l(NULL), r(NULL) {};
 
};
typedef Item* pitem;
 
// prints the in-order traversal of a tree
void output(Item *t) {
    if (!t) return;
    output(t -> l);
    cout << t -> key;
    output(t -> r);
}
 
inline int size(const pitem &item) { return item ? item -> size : 0; }
 
void split(pitem t, int key, pitem &l, pitem &r) {
    if (!t) {
        l = r = NULL; 
        return;
    }
    if (size(t -> l) < key) {
        split(t->r, key - size(t -> l) - 1, t->r, r);
        l = t;
    }
    else {
        split(t->l, key, l, t->l);
        r = t;
    }
    t -> size = 1 + size(t -> l) + size(t -> r);
}
 
// merge left and right into t
void merge(pitem &t, pitem l, pitem r) {
    if (!l || !r) {
        t = l ? l : r;
        return;
    }
    else if (l -> prior > r -> prior) {
        merge(l -> r, l -> r, r), t = l;
    }
    else {
        merge(r -> l, l, r -> l), t = r;
    }
    t -> size = 1 + size(t -> l) + size(t -> r);
}

int N, M;
 
void solve() {
    pitem root = nullptr;
    string S;
    cin >> N >> M >> S;
    for (char ch : S) {
        merge(root, root, new Item(ch));
    }
    for (int i = 0; i < M; i++) {
        int a, b;
        cin >> a >> b;
        a--; b--;
        pitem t0 = nullptr, t1 = nullptr, t2 = nullptr, t3 = nullptr;
        split(root, a, t0, t1);
        split(t1, b - a + 1, t2, t3);
        merge(root, t0, t3);
        merge(root, root, t2);
    }
    output(root);
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Substring Reversals

```cpp
struct Item {
    char key;
    int prior, size;
    bool rev;
    Item *l, *r;
    Item() {};
    Item(char key) : key(key), prior(rand()), size(1), l(NULL), r(NULL) {};
    Item(char key, int prior) : key(key), prior(prior), size(1), l(NULL), r(NULL) {};
 
};
typedef Item* pitem;

void propagate(pitem t) {
    if (t && t -> rev) {
        t -> rev = false;
        swap(t -> l, t -> r);
        if (t -> l) t -> l -> rev ^= true;
        if (t -> r) t -> r -> rev ^= true;
    }
}
 
// prints the in-order traversal of a tree
void output(Item *t) {
    if (!t) return;
    propagate(t);
    output(t -> l);
    cout << t -> key;
    output(t -> r);
}
 
int size(const pitem &item) { return item ? item -> size : 0; }
 
void split(pitem t, pitem &l, pitem &r, int val) {
    if (!t) {
        l = r = NULL; 
        return;
    }
    propagate(t);
    if (size(t -> l) < val) {
        split(t->r, t->r, r, val - size(t -> l) - 1);
        l = t;
    }
    else {
        split(t->l, l, t->l, val);
        r = t;
    }
    t -> size = 1 + size(t -> l) + size(t -> r);
}
 
// merge left and right into t
void merge(pitem &t, pitem l, pitem r) {
    if (!l || !r) {
        t = l ? l : r;
        return;
    }
    propagate(l);
    propagate(r);
    if (l -> prior > r -> prior) {
        merge(l -> r, l -> r, r), t = l;
    }
    else {
        merge(r -> l, l, r -> l), t = r;
    }
    t -> size = 1 + size(t -> l) + size(t -> r);
}

void reverse(pitem t, int l, int r) {
    pitem t1, t2, t3;
    split (t, t1, t2, l);
    split(t2, t2, t3, r - l + 1);
    t2 -> rev ^= true;
    merge(t, t1, t2);
    merge(t, t, t3);
}
 
int N, M;
 
void solve() {
    pitem root = nullptr;
    string S;
    cin >> N >> M >> S;
    for (char ch : S) {
        merge(root, root, new Item(ch));
    }
    for (int i = 0; i < M; i++) {
        int a, b;
        cin >> a >> b;
        a--; b--;
        reverse(root, a, b);
    }
    output(root);
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Reversals and Sums

```cpp
struct Item {
    int val, prior, size, sum;
    bool rev;
    Item *l, *r;
    Item() {};
    Item(int val) : val(val), prior(rand()), size(1), l(NULL), r(NULL) {};
    Item(int val, int prior) : val(val), prior(prior), size(1), l(NULL), r(NULL) {};
};
typedef Item* pitem;

void push(pitem t) {
    if (t && t -> rev) {
        t -> rev = false;
        swap(t -> l, t -> r);
        if (t -> l) t -> l -> rev ^= true;
        if (t -> r) t -> r -> rev ^= true;
    }
}

int size(const pitem &t) { return t ? t -> size : 0; }
int sum(const pitem &t) { return t ? t -> sum : 0; }

// prints the in-order traversal of a tree
void output(Item *t) {
    if (!t) return;
    push(t);
    output(t -> l);
    cout << t -> val << " ";
    output(t -> r);
}

void pull(pitem t) {
    if (t) {
        push(t -> l); push(t -> r);
        t -> size = 1 + size(t -> l) + size(t -> r);
        t -> sum = t -> val + sum(t -> l) + sum(t -> r);
    }
}
  
void split(pitem t, pitem &l, pitem &r, int val) {
    push(t);
    if (!t) {
        l = r = NULL; 
    } else if (size(t -> l) < val) {
        split(t->r, t->r, r, val - size(t -> l) - 1);
        l = t;
    }
    else {
        split(t->l, l, t->l, val);
        r = t;
    }
    pull(t);
}
 
// merge left and right into t
void merge(pitem &t, pitem l, pitem r) {
    push(l); push(r);
    if (!l || !r) {
        t = l ? l : r;
    } else if (l -> prior > r -> prior) {
        merge(l -> r, l -> r, r), t = l;
    }
    else {
        merge(r -> l, l, r -> l), t = r;
    }
    pull(t);
}

void reverse(pitem t, int l, int r) {
    pitem t1, t2, t3;
    split (t, t1, t2, l);
    split(t2, t2, t3, r - l + 1);
    t2 -> rev ^= true;
    merge(t, t1, t2);
    merge(t, t, t3);
}

int sum(pitem t, int l, int r) {
    pitem t1, t2, t3;
    split(t, t1, t2, l);
    split(t2, t2, t3, r - l + 1);
    int ans = sum(t2);
    merge(t, t1, t2);
    merge(t, t, t3);
    return ans;
}
 
int N, M;
 
void solve() {
    pitem root = nullptr;
    cin >> N >> M;
    for (int i = 0; i < N; i++) {
        int x;
        cin >> x;
        merge(root, root, new Item(x));
    }
    for (int i = 0; i < M; i++) {
        int t, a, b;
        cin >> t >> a >> b;
        a--; b--;
        if (t == 1) {
            reverse(root, a, b);
        } else {
            cout << sum(root, a, b) << endl;
        }
    }
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'

struct Item {
    int key, prior;
    Item *l, *r;
    Item() {};
    Item(int key) : key(key), prior(rand()), l(NULL), r(NULL) {};
    Item(int key, int prior) : key(key), prior(prior), l(NULL), r(NULL) {};

};
typedef Item* pitem;

void dump(pitem t, int depth = 0) {
    if (!t) return;
    dump(t->l, depth + 1);
    cout << t->key << ' ' << depth << endl;
    dump(t->r, depth + 1);
}

void split(pitem t, int key, pitem &l, pitem &r) {
    if (!t) {
        l = r = NULL; return;
    }
    cout << "=================================" << endl;
    cout << t->key << endl;
    dump(t);
    if (t->key <= key) {
        split(t->r, key, t->r, r);
        cout << "less than or equal" << endl;
        dump(t);
        cout << "left" << endl;
        dump(l);
        cout << "right" << endl;
        dump(r);
        cout << "=================================" << endl;
        l = t;
    }
    else {
        split(t->l, key, l, t->l);
        cout << "greater than" << endl;
        dump(t);
        cout << "left" << endl;
        dump(l);
        cout << "right" << endl;
        dump(r);
        cout << "=================================" << endl;
        r = t;
    } 
}



int N;

void solve() {
    pitem root = new Item(30, 12);
    // pitem root = node;
    root->l = new Item(20, 11);
    root->r = new Item(40, 9);
    root->l->l = new Item(5, 6);
    root->l->r = new Item(25, 10);
    root->l->r->l = new Item(22, 1);
    root->l->r->r = new Item(27, 2);
    root->r->r = new Item(50, 8);
    root->r->r->r = new Item(60, 5);
    root->r->r->l = new Item(55, 4);
    root->r->l = new Item(35, 7);
    root->r->l->r = new Item(38, 3);
    pitem left = nullptr, right = nullptr;
    split(root, 35, left, right);
    dump(left);
    for (int i = 0; i < N; i++) {
        int a, b;
        cin >> a >> b;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}

```