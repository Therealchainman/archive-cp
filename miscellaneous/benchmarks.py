import timeit

def generateString(s):
    return s*1000

# using any
def findAnagramsAny(s, p):
    result, freq = [], [0]*26
    m = len(p)
    def getInt(ch):
        return ord(ch)-ord('a')
    for c in p:
        freq[getInt(c)]+=1
    for i in range(len(s)):
        freq[getInt(s[i])]-=1
        if i>=m:
            freq[getInt(s[i-m])]+=1
        if not any(f!=0 for f in freq):
            result.append(i-m+1)
    return result

# using all
def findAnagramsAll(s, p):
    result, freq = [], [0]*26
    m = len(p)
    def getInt(ch):
        return ord(ch)-ord('a')
    for c in p:
        freq[getInt(c)]+=1
    for i in range(len(s)):
        freq[getInt(s[i])]-=1
        if i>=m:
            freq[getInt(s[i-m])]+=1
        if all(f==0 for f in freq):
            result.append(i-m+1)
    return result

if __name__=='__main__':
    s, p = generateString("cbaebabacd"), "abc"
    print(f"Time elapsed for 1000 executions of the function that uses ALL: {timeit.timeit(lambda: findAnagramsAll(s,p), number=1000)} seconds")
    print(f"Time elapsed for 1000 executions of the function that uses ANY: {timeit.timeit(lambda: findAnagramsAny(s,p), number=1000)} seconds")