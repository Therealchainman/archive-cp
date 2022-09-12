## Z Algorithm

```py
def main():
    text = input()
    pat = input()
    sentinel_char = '$'
    s = pat + sentinel_char + text
    sLen = len(s)
    patLen = len(pat)
    z = [0]*sLen
    left=right=0
    for i in range(1,sLen):
        if i>right:
            left=right=i
            while right<sLen and s[right-left]==s[right]:
                right+=1
            z[i]=right-left
            right-=1
        else:
            k=i-left
            if z[k]<right-i+1:
                z[i]=z[k]
            else:
                left=i
                while right<sLen and s[right-left]==s[right]:
                    right+=1
                z[i]=right-left
                right-=1
    return z.count(patLen)
if __name__ == '__main__':
    print(main())
```