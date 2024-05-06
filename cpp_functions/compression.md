# Compression

## Coordinate Compression

this is 1-indexed compression, 

```cpp
sort(values.begin(), values.end());
auto it = unique(values.begin(), values.end()); 
values.resize(distance(values.begin(), it));
map<int, int> compress;
for (int &v : values) {
    compress[v] = compress.size() + 1;
}
```

This is how to dedupe vector

```cpp
sort(poss.begin(), poss.end());
poss.erase(unique(poss.begin(), poss.end()), poss.end());
```