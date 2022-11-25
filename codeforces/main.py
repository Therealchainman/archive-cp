def main():
    s = input()
    n = len(s)
    arr = ['a']*(2*n)
    for i, ch in enumerate(s):
        arr[i] = arr[~i] = ch
    return ''.join(arr)

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        print(main())