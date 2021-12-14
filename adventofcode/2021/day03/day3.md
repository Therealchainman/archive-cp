Python 

# Part 1

First solution for this problem 

```py
binaryArr = f.read().splitlines()
n = len(binaryArr)
nBinaryDigits = len(binaryArr[0])
mostCommon = "".join(map(str, map(lambda index: 1 if sum(1 for binary in binaryArr if binary[index]=='1')>=n//2 else 0, range(nBinaryDigits))))
leastCommon = "".join(map(str, map(lambda index: 1 if sum(1 for binary in binaryArr if binary[index]=='1')<=n//2 else 0, range(nBinaryDigits))))
print(int(mostCommon,2)*int(leastCommon,2))
```

Second solution for this problem using the the zip to get all the corresponding bits together and perform a count on them as a tuple.  And use that to predict if it is a 1 or 0, and then use xor to flip all the bits. 

```py
gammaBits = list(map(lambda x: '1' if x.count('1')>x.count('0') else '0', zip(*f.read().splitlines())))
gamma = int("".join(gammaBits), 2)
n = len(gammaBits)
epsilon = gamma ^ ((1<<n)-1)
print(gamma*epsilon)
```


```c++
int convBinaryToDecimal(string& s) {
    int n = s.size(), dec = 0;
    for (int i = 0; i < n; i++) {
        dec += ((s[i] - '0')*(1<<(n-i-1)));
    }
    return dec;
}
int main() {
    freopen("inputs/input.txt", "r", stdin);
    vector<string> binaryArr;
    string input;
    while (getline(cin, input)) {
        binaryArr.push_back(input);
    }
    int n = binaryArr[0].size();
    vector<string> arr;
    for (int i = 0;i<n;i++) {
        string s = "";
        for (int j = 0;j<binaryArr.size();j++) {
            s += binaryArr[j][i];
        }
        arr.push_back(s);
    }
    string gamm = "";
    for (int i = 0;i<n;i++) {
        gamm += (count(arr[i].begin(), arr[i].end(), '1') > count(arr[i].begin(), arr[i].end(), '0') ? '1' : '0');
    }
    int gamma = convBinaryToDecimal(gamm);
    int epsilon = gamma ^ ((1<<n)-1);
    cout << gamma*epsilon << endl;
}
```

# Part 2

```py
binaryArr = f.read().splitlines()
oxygensArr = binaryArr
nBinaryDigits = len(binaryArr[0])
for index in range(nBinaryDigits): 
    cnt = sum(1 if oxy[index]=='1' else -1 for oxy in oxygensArr)
    oxygensArr = list(filter(lambda binary: binary[index]=='1' if cnt>=0 else binary[index]=='0', oxygensArr))
    if len(oxygensArr)==1:
        break  
co2Arr = binaryArr
for index in range(nBinaryDigits):
    cnt = sum(1 if co2[index]=='1' else -1 for co2 in co2Arr)
    co2Arr = list(filter(lambda binary: binary[index]=='0' if cnt>=0 else binary[index]=='1', co2Arr))
    if len(co2Arr)==1:
        break
oxygenVal = int(oxygensArr[0], 2)
co2Val = int(co2Arr[0], 2)
print(oxygenVal*co2Val)
```

Second solution using multiple function to get the oxygen generator rating and co2 scrubber rating to get the final rating. 
This uses a Counter and most_common() on it to get the most common value, which needs to check for tiebreakers and
needs to return the lowest if it is co2.  

```py
from collections import Counter

"""
Reduces the size of the array until it is 1, and that is the value after convering to int
"""
def getOxygenGeneratorRating(ratingArr):
    nBinaryDigits = len(ratingArr[0])
    for index in range(nBinaryDigits):
        cands = Counter(list(zip(*ratingArr))[index]).most_common()
        mostCommon = '1' if len(cands) == 2 and cands[0][1] == cands[1][1] else cands[0][0]
        ratingArr = list(filter(lambda x: x[index] == mostCommon, ratingArr))
        if len(ratingArr) == 1:
            break
    return int(ratingArr[0], 2)

def getCO2ScrubberRating(ratingArr):
    nBinaryDigits = len(ratingArr[0])
    for index in range(nBinaryDigits):
        cands = Counter(list(zip(*ratingArr))[index]).most_common()
        leastCommon = '0' if len(cands) == 2 and cands[0][1] == cands[1][1] else cands[-1][0]
        ratingArr = list(filter(lambda x: x[index] == leastCommon, ratingArr))
        if len(ratingArr) == 1:
            break
    return int(ratingArr[0], 2)    

if __name__ == '__main__':
    with open("inputs/input.txt", "r") as f:
        binaryArr = f.read().splitlines();
        print(getOxygenGeneratorRating(binaryArr)*getCO2ScrubberRating(binaryArr))
```

Third solution using a recursive algorithm 

```py
def getOxygenGeneratorRating(index, ratingArr):
    if len(ratingArr) == 1:
        return int(ratingArr[0], 2)
    nBinaryDigits = len(ratingArr[0])
    if index == nBinaryDigits:
        return -1 # error
    cands = Counter(list(zip(*ratingArr))[index]).most_common()
    mostCommon = '1' if len(cands) == 2 and cands[0][1] == cands[1][1] else cands[0][0]
    return getOxygenGeneratorRating(index + 1, list(filter(lambda x: x[index] == mostCommon, ratingArr)))

def getCO2ScrubberRating(index, ratingArr):
    if len(ratingArr) == 1:
        return int(ratingArr[0], 2)
    nBinaryDigits = len(ratingArr[0])
    if index == nBinaryDigits:
        return -1 # error
    cands = Counter(list(zip(*ratingArr))[index]).most_common()
    leastCommon = '0' if len(cands) == 2 and cands[0][1] == cands[1][1] else cands[-1][0]
    return getCO2ScrubberRating(index + 1, list(filter(lambda x: x[index] == leastCommon, ratingArr)))  

if __name__ == '__main__':
    with open("inputs/input.txt", "r") as f:
        binaryArr = f.read().splitlines();
        print(getOxygenGeneratorRating(0, binaryArr)*getCO2ScrubberRating(0, binaryArr))
```

Fourth solution reducing to a single rating function

```py
from collections import Counter
def getRating(index, ratingArr, indicator):
    if len(ratingArr) == 1:
        return int(ratingArr[0], 2)
    nBinaryDigits = len(ratingArr[0])
    if index == nBinaryDigits:
        return -1 # error
    cands = Counter(list(zip(*ratingArr))[index]).most_common()
    interest = str(1+indicator) if len(cands) == 2 and cands[0][1] == cands[1][1] else cands[indicator][0]
    return getRating(index + 1, list(filter(lambda x: x[index] == interest, ratingArr)), indicator)

if __name__ == '__main__':
    with open("inputs/input.txt", "r") as f:
        binaryArr = f.read().splitlines();
        print(getRating(0, binaryArr, 0)*getRating(0, binaryArr, -1))
```

```c++
int convBinaryToDecimal(string& s) {
    int n = s.size(), dec = 0;
    for (int i = 0; i < n; i++) {
        dec += ((s[i] - '0')*(1<<(n-i-1)));
    }
    return dec;
}
int getRating(int index, vector<string>& ratingArr, int indicator) {
    if (ratingArr.size()==1) {
        return convBinaryToDecimal(ratingArr[0]);
    }
    int n = ratingArr[0].size();
    if (index == ratingArr[0].size()) {
        return -1;
    }
    vector<string> arr;
    for (int i = 0; i < n; i++) {
        string s = "";
        for (int j = 0; j < ratingArr.size(); j++) {
            s += ratingArr[j][i];
        }
        arr.push_back(s);
    }
    int cnt1 = count(arr[index].begin(), arr[index].end(), '1'), cnt0 = count(arr[index].begin(), arr[index].end(), '0');
    vector<string> tmp;
    for (int i = 0;i<ratingArr.size();i++) {
        if (indicator) {
            char interest = (cnt1>=cnt0)?'1':'0';
            if (ratingArr[i][index] == interest) {
                tmp.push_back(ratingArr[i]);
            }
        } else {
            char interest = (cnt1<cnt0) ? '1' : '0';
            if (ratingArr[i][index] == interest) {
                tmp.push_back(ratingArr[i]);
            }
        }
    }
    return getRating(index + 1, tmp, indicator);
}
int main() {
    freopen("inputs/input.txt", "r", stdin);
    vector<string> binaryArr;
    string input;
    while (getline(cin, input)) {
        binaryArr.push_back(input);
    }
    cout << getRating(0, binaryArr, 1)*getRating(0, binaryArr, 0) << endl;
}
```