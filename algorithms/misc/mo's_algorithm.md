# Mo's Algorithm


This is the general outline of the algorithm.  It is brilliant for problems where you need to get answer for many segments or intervals.  It finds the answer for each current window in optimal way.  The only requirement is that the remove and add function need to be relatively fast about O(1) for it to be fast enough. 

Time complexity is roughly $O\big(NF\sqrt{N}\big)$, where F is time complexity of add and remove function

This is when you pick $B = \frac{N}{\sqrt{Q}}$.

```cpp
int block_size;

struct Query {
    int l, r, idx;

    bool operator<(const Query &other) const {
        int b1 = l / block_size, b2 = other.l / block_size;
        if (b1 != b2) return b1 < b2;
        if (b1 & 1) return r > other.r;
        return r < other.r;
    }
};

void remove(idx);  // TODO: remove value at idx from data structure
void add(idx);     // TODO: add value at idx from data structure
int getAnswer();  // TODO: extract the current answer of the data structure

vector<int> mo_s_algorithm(vector<Query> queries) {
    block_size = max(1, (int)(N / max(1.0, sqrt(Q))));
    vector<int> answers(Q);
    sort(queries.begin(), queries.end());

    // TODO: initialize data structure
    int curL = 0, curR = -1;
    for (const Query& q : queries) {
        while (curL > q.l) add(--curL);
        while (curR < q.r) add(++curR);
        while (curL < q.l) { remove(curL); ++curL; }
        while (curR > q.r) { remove(curR); --curR; }
        answers[q.idx] = getAnswer();
    }
    return answers;
}
```