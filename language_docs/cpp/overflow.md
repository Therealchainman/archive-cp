# Overflow

## How to determine if multiplication of two numbers will overflow

In here int is an alias for long long data type, so this is a 64 bit signed integer.

```cpp
bool is_overflow(int x, int y) {
    int result;
    return __builtin_mul_overflow(x, y, &result);
}
```