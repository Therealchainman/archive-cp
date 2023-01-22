# palindrome algorithms

## Check if current string is palindrome in linear time

```py
def is_palindrome(part: str) -> bool:
    left, right = 0, len(part) - 1
    while left < right and part[left] == part[right]:
        left += 1
        right -= 1
    return left >= right
```