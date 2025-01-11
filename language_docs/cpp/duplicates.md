# Duplicates

A cool way to return boolean to indicate if a sorted container contains duplicates or not. 

```cpp
bool has_dupes = adjacent_find(arr.begin(), arr.end()) != arr.end()
```