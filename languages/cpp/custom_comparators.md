# Custom Comparators


## Priority Queue

You are comparing a and b, and if the operator returns true than a comes after b, order is (b, a)

```cpp
struct priorityComp {
    // returns true mean it comes before in weak ordering, but means it comes after in priority queue.
    // So return false means it comes before in priority queue.
    bool operator()(const pair<int, int>& a, const pair<int, int>& b) const {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    }
};
```

MIN HEAP EXAMPLE

```cpp
struct stateComp {
    bool operator()(const State& a, const State& b) const {
        // min heap, if a.cost <= b.cost than it returns false, and it knows the order is (a, b)
        return a.cost > b.cost;
    }
};

priority_queue<State, vector<State>, stateComp> maxheap;
```

## Vector 

sorting edges containing object Edge in descending order of w variable

```cpp
    sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.w > b.w; // Descending order
    })
```

## Custom sort in the struct

sorts for w + h / (wh) < other.w + other.h / (other.wh)

```cpp
struct Rect {
    int w, h;
    Rect() {}
    Rect(int w, int h) : w(w), h(h) {}
    bool operator<(const Rect &other) const {
        return (w + h) * other.w * other.h > (other.w + other.h) * w * h;
    }
};
```
