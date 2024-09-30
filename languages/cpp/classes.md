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