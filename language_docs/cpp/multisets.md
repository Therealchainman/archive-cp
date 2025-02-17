# multisets

Useful functions for multisets

## MAX

Get the maximum value in multiset or 0 if doesn't exist, or change to any desired default when multiset is empty

```cpp
int maximum(multiset<int>& ms) {
    auto max_val_iterator = ms.rbegin();
    return max_val_iterator != ms.rend() ? *max_val_iterator : 0;
}
```

## Erase One Element

Erase just one element from multiset and not all the elements with the same value.  Just be cautious if you call erase with an iterator it removes one instance of that value.  But if you use erase with the value it removes all instances of that value. 

```cpp
void erase(multiset<int>& s, int x) {
	s.erase(s.find(x));
}
```

## LOWER BOUND 

```cpp
auto it = s.lower_bound(x);
if (it != s.end()) {
    // do something with it
    s.erase(it);
}
```

## Checking for containment

Do not use count(x) with multiset, it will return the number of elements with value x. Instead, use find(x) != end(), it is much faster.  That is count() does not run in O(logn) time, as it does for set.  So use find() which does run in O(logn) time.
