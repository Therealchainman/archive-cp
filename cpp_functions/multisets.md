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

Erase just one element from multiset and not all the elements with the same value

```cpp
void erase(multiset<int>& s, int x) {
	s.erase(s.find(x));
}
```