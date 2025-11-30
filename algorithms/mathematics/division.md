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

## Special case

In some scenarios with I think really large integers or negatives and what not, you want to use these floor and ceil division functions. 

Cause there is an issue with negative numbers and large numbers. 

```cpp
static inline int128 floor(int128 a, int128 b) {
    // b must not be zero
    int128 q = a / b;
    int128 r = a % b;
    if (r != 0 && ((r > 0) != (b > 0))) --q;
    return q;
}

// ceil division that matches mathematical ceil for integers
static inline int128 ceil(int128 a, int128 b) {
    // b must not be zero
    return -floor(-a, b);
}
```