# Division

## floor division

Safer to use x // y instead of math.floor(x / y)

This is the most accurate floor division that handles negative numbers correctly.

```cpp
int64 floor(int64 x, int64 y) {
    return (x / y) - ((x % y) != 0 && (x < 0) != (y < 0));
}
```

## ceil division

safter to use (x + y - 1) // y instead of math.ceil(x / y).  

```cpp
int64 ceil(int64 x, int64 y) {
    return (x + y - 1) / y;
}
```