# Advent of Code 2021

## Day 1

### Part 1

Iterate over input

```c++
const int INF = 1e8;
int main() {
    freopen("inputs/input1.txt", "r", stdin);
    freopen("outputs/output1.txt", "w", stdout);
    string tmp;
    int cnt = 0, depth, pDepth = INF;
    while(getline(cin, tmp)) {
        depth = stoi(tmp);
        cnt += (depth>pDepth);
        pDepth = depth;
    }
    cout<<cnt<<endl;
}
```

```py
"""
part 1
"""
import sys
if __name__ == "__main__":
    sys.stdout = open('outputs/output1.txt', 'w')
    with open("inputs/input1.txt", "r") as f:
        data = list(map(int,f.read().splitlines()))
        print(sum(1 for prev, num in zip(data, data[1:]) if num>prev))
    sys.stdout.close()
```

### Part 2

compare the elements

```c++
int main() {
    freopen("inputs/input1.txt", "r", stdin);
    freopen("outputs/output2.txt", "w", stdout);
    string tmp;
    int cnt = 0, depth;
    vector<int> depths;
    while(getline(cin, tmp)) {
        depth = stoi(tmp); 
        depths.push_back(depth);
    }
    for(int i=3; i<depths.size(); i++) {
        cnt += (depths[i]>depths[i-3]);
    }
    cout<<cnt<<endl;
}
```


```py
if __name__ == "__main__":
    with open("inputs/input1.txt", "r") as f:
        data = list(map(int, f.read().splitlines()))
        print(sum(1 for num1, num4 in zip(data, data[3:]) if num4>num1))
```

## Day 2

### Part 2: 

```py
from collections import namedtuple
if __name__ == '__main__':
    with open("inputs/input1.txt", "r") as f:
        commands = namedtuple('command', ['direction', 'magnitude'])
        arr = map(lambda x: commands(x.split()[0], int(x.split()[1])), f.read().splitlines())
        hor, depth = 0, 0
        for command in arr:
            if command.direction=="forward":
                hor += command.magnitude
            elif command.direction=="up":
                depth -= command.magnitude
            elif command.direction=="down":
                depth += command.magnitude
        print(hor*depth)
```

Improved solution using functional programming with sum and map.
```py
if __name__ == '__main__':
    with open("inputs/input1.txt", "r") as f:
        arr = list(map(lambda x: (x.split()[0], int(x.split()[1])), f.read().splitlines()))
        hor = sum(magnitude for direction, magnitude in arr if direction in ['forward'])
        depth = sum(magnitude*(1 if direction=='down' else -1) for direction, magnitude in arr if direction in ['up', 'down'])
        print(hor*depth)
```

```c++
int main() {
    freopen("inputs/input1.txt", "r", stdin);
    freopen("outputs/output1.txt", "w", stdout);
    string input;
    long long depth = 0, hor = 0;
    while (getline(cin, input)) {
        int pos = input.find(" ");
        string direction = input.substr(0, pos);
        int magnitude = stoi(input.substr(pos + 1));
        if (direction == "down") {
            depth += magnitude;
        } else if (direction == "up") {
            depth -= magnitude;
        } else if (direction == "forward") {
            hor += magnitude;
        }
    }
    cout<<depth*hor<<endl;
}
```

```py
if __name__ == '__main__':
    with open("inputs/input1.txt", "r") as f:
        arr = map(lambda x: (x.split()[0], int(x.split()[1])), f.read().splitlines())
        hor, depth, aim = 0, 0, 0
        for dir, magnitude in arr:
            if dir=="forward":
                hor += magnitude
                depth += aim*magnitude
            elif dir=="up":
                aim -= magnitude
            elif dir=="down":
                aim += magnitude
        print(hor*depth)
```

```py
if __name__ == '__main__':
    with open("inputs/input1.txt", "r") as f:
        arr = list(map(lambda x: (x.split()[0], int(x.split()[1])), f.read().splitlines()))
        hor, depth, aim = 0, 0, 0
        for dir, magnitude in arr:
            hor += magnitude if dir == 'forward' else 0
            depth += (magnitude*aim) if dir == 'forward' else 0
            aim += magnitude *(1 if dir == 'down' else -1 if dir=='up' else 0)
        print(hor*depth)
```

```c++
int main() {
    freopen("inputs/input1.txt", "r", stdin);
    freopen("outputs/output2.txt", "w", stdout);
    string input;
    long long depth = 0, hor = 0, aim = 0;
    while (getline(cin, input)) {
        int pos = input.find(" ");
        string direction = input.substr(0, pos);
        int magnitude = stoi(input.substr(pos + 1));
        if (direction == "down") {
            aim += magnitude;
        } else if (direction == "up") {
            aim -= magnitude;
        } else if (direction == "forward") {
            hor += magnitude;
            depth += (magnitude*aim);
        }
    }
    cout<<depth*hor<<endl;
}
```

## Day 3

### Part 1

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

### Part 2

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

## Day 4

Python

### Part 1

Brute force solution to find the first board that wins.  

```py
def winningScore(game):
    for i in range(5):
        markedRow = sum(1 for j in range(5) if game[i][j] < 0)
        if markedRow==5:
            return sum(-game[i][j] for j in range(5))
        markedCol = sum(1 for j in range(5) if game[j][i] < 0)
        if markedCol==5:
            return sum(-game[j][i] for j in range(5))
    return -1
if __name__ == '__main__':
    with open("inputs/input.txt", "r") as f:
        calls = list(map(lambda x: int(x.replace('\n', '')), f.readline().split(",")))
        boards = []
        while f.readline():
            boards.append([list(map(int, f.readline().split())) for _ in range(5)])
        def bingoCalls():
            for num in calls:
                for board in boards:
                    for i in range(5):
                        for j in range(5):
                            if board[i][j] == num:
                                board[i][j] = -num
                    score = winningScore(board)
                    if score != -1:
                        unMarkedScore = sum(board[i][j] for i in range(5) for j in range(5) if board[i][j] > 0)
                        return unMarkedScore*num
        print(bingoCalls())
```

## Day 5

### Part 1

```py
from collections import namedtuple
with open("inputs/input.txt", "r") as f:
    Point = namedtuple("Point", ["x", "y"])
    data = f.read().splitlines()
    data = [x.split(" -> ") for x in data]
    data = [(Point(int(x[0].split(",")[0]), int(x[0].split(",")[1])), Point(int(x[1].split(",")[0]), int(x[1].split(",")[1]))) for x in data]
    matrix = [[0 for _ in range(1000)] for _ in range(1000)]
    for start, end in data:
        if start.x!=end.x and start.y!=end.y:
            continue
        deltaX = 1 if end.x>start.x else -1 if end.x<start.x else 0
        deltaY = 1 if end.y>start.y else -1 if end.y<start.y else 0
        ix, iy = start.x, start.y
        while ix != end.x or iy != end.y:
            matrix[ix][iy] += 1
            ix += deltaX
            iy += deltaY
        matrix[end.x][end.y] += 1
    cnt = sum(1 for x in range(len(matrix)) for y in range(len(matrix[0])) if matrix[x][y]>=2)
    print(cnt)
```

### Part 2

vector of pair<int,int> to represent points and map of pair<int,int> to represent frequency

```c++
int main() {
    freopen("inputs/input.txt", "r", stdin);
    string line;
    vector<vector<pair<int,int>>> lineSegments;
    while (getline(cin, line)) {
        int pos = line.find(" -> ");
        int pos1 = line.find(',');
        int pos2 = line.find(',', pos1 + 1);
        string startx = line.substr(0, pos1), starty = line.substr(pos1 + 1, pos - pos1- 1), endx = line.substr(pos + 4, pos2-(pos+3)-1), endy = line.substr(pos2+1);
        lineSegments.push_back({{stoi(startx), stoi(starty)}, {stoi(endx), stoi(endy)}});
    }
    map<pair<int,int>, int> freq;
    for (auto &lineSegment : lineSegments) {
        auto start = lineSegment[0], end = lineSegment[1];
        #define x first
        #define y second
        int deltaX = end.x > start.x ? 1 : end.x < start.x ? -1 : 0;
        int deltaY = end.y > start.y ? 1 : end.y < start.y ? -1 : 0;
        for (int ix = start.x, iy = start.y; ix != end.x || iy != end.y; ix += deltaX, iy += deltaY) {
            freq[{ix, iy}]++;
        }
        freq[{end.x, end.y}]++;
    }
    int cnt = 0;
    for (auto point: freq) {
        #define count second
        cnt += (point.count >= 2);
    }
    cout<<cnt<<endl;
}
```

namedtuples and using a method to iterate through diagonals, verticals, and horizontals. 
```py
from collections import namedtuple
with open("inputs/input.txt", "r") as f:
    Point = namedtuple("Point", ["x", "y"])
    data = f.read().splitlines()
    data = [x.split(" -> ") for x in data]
    data = [(Point(int(x[0].split(",")[0]), int(x[0].split(",")[1])), Point(int(x[1].split(",")[0]), int(x[1].split(",")[1]))) for x in data]
    matrix = [[0 for _ in range(1000)] for _ in range(1000)]
    for start, end in data:
        deltaX = 1 if end.x>start.x else -1 if end.x<start.x else 0
        deltaY = 1 if end.y>start.y else -1 if end.y<start.y else 0
        ix, iy = start.x, start.y
        while ix != end.x or iy != end.y:
            matrix[ix][iy] += 1
            ix += deltaX
            iy += deltaY
        matrix[end.x][end.y] += 1
    cnt = sum(1 for x in range(len(matrix)) for y in range(len(matrix[0])) if matrix[x][y]>=2)
    print(cnt)
```

## Day 6

### Part 1

This solution works for 80 days, it uses an array to keep track of the number of lanternfish with days left because only 8 days so that is a small array.  

```c++
const int N = 80;
long long lanternFish[9];
int main() {
    freopen("inputs/input.txt", "r", stdin);
    memset(lanternFish, 0, sizeof(lanternFish));
    string input, tmp;
    cin>>input;
    stringstream ss(input);
    while (getline(ss, tmp, ',')) {
        lanternFish[stoll(tmp)]++;
    }
    for (int day = 0;day<N;day++) {
        long long born = lanternFish[0];
        for (int i = 0;i<8;i++) {
            lanternFish[i] = lanternFish[i+1];
        }
        lanternFish[6] += born;
        lanternFish[8] = born;
    }
    long long cnt = accumulate(lanternFish, lanternFish+9, 0LL);
    cout<<cnt<<endl;
```

### Part 2

Change N  to 256

## Day 7


### Part 1 

Brute force solution with time of O(nm), where n = len(positions) and m = mx-mn

```py
positions = list(map(int, f.read().split(',')))
mx, mn = max(positions), min(positions)
lo, hi, initial = mn, mx, 1458460
minFuel = reduce(lambda minFuel, pos: min(minFuel, sum(abs(pos - crabPos) for crabPos in positions)), range(lo, hi + 1), initial)
print(minFuel)
```

binary search solution with O(nlog(m)) time

```py
positions = list(map(int, f.read().split(',')))
lo, hi = min(positions), max(positions)
def fuelCost(target):
    return sum(abs(target-pos) for pos in positions)
while lo < hi:
    mid = (lo + hi + 1) >> 1
    if fuelCost(mid) < fuelCost(mid-1):
        lo = mid
    else:
        hi = mid - 1
print(fuelCost(lo))
```
Solution using median, this is done in O(n) time
```py
positions = list(map(int, f.read().split(',')))
medianPos = int(median(positions))
print(sum(abs(medianPos - pos) for pos in positions))
```
Using numpy for solution 
```py
positions = list(map(int, f.read().split(',')))
minFuel = int(abs(positions - np.median(positions)).sum())
print(minFuel)
```
### Part 2


```py
positions = list(map(int, f.read().split(',')))
mx, mn = max(positions), min(positions)
lo, hi, initial = mn, mx, 1161136310
minFuel = reduce(lambda minFuel, pos: min(minFuel, sum(abs(pos-crabPos)*(abs(pos-crabPos)+1)//2 for crabPos in positions)), range(lo, hi + 1), initial)
print(minFuel)
```


```py
positions = list(map(int, f.read().split(',')))
hi, lo = max(positions), min(positions)
def fuelCost(target):
    return sum(abs(target-pos)*(abs(target-pos)+1)//2 for pos in positions)
while lo < hi:
    mid = (lo + hi + 1) >> 1
    if fuelCost(mid) < fuelCost(mid-1):
        lo = mid
    else:
        hi = mid - 1
print(fuelCost(lo))
```
Using mean to solve the problem, in O(n) time using statistics
```py
positions = list(map(int, f.read().split(',')))
meanPos = int(mean(positions))
minFuel = sum(abs(pos-meanPos)*(abs(pos-meanPos)+1)//2 for pos in positions)
print(minFuel)
```

Using mean in numpy 

```py
positions = np.array(list(map(int, f.read().split(','))))
minFuel = int(sum(n*(n+1)/2 for n in abs(positions - int(np.mean(positions)))))
print(minFuel)
```

## Day 8

### Part 1

We just need to count the digits that have a unique number of on segments.  This is 2,3,4,7.
I can create a set and then just count the number of these that have that size int he output of the patterns


```py
A = f.read().splitlines()
sizeDigitMap = {2,4,3,7}
outputs = [x[x.find('|')+2:].split() for x in A]
cnt = sum(1 for out in outputs for pat in out if len(pat) in sizeDigitMap)
print(cnt)
```

### Part 2
This problem can be solved a couple ways.  One way is with permutations 
Still need to explore the permutation method. 
DFS + backtracking algorithm with OOP
So I use the fact that there are a certain number of on segments that need to be on in common for 
a pattern to be valid.  Such as maybe 1 and 2 need to always have 2 segments in common. so given each 
alphabet character represents a specific segment being on.  We don't know which segment. But with this
comparison between them we can say for certain if a having a character assigned to a segment is appropriate. 

Another way to approach this would be with generating all permutations of having characters
represent certain segments.  Then just need to check that it works as well.  

```py
class SevenSegmentSearch:
    def __init__(self):
        self.baseMap = {0 : 'abcefg', 1: 'cf', 2: 'acdeg', 3: 'acdfg', 4: 'bcdf', 5: 'abdfg', 6: 'abdefg',
        7: 'acf', 8: 'abcdefg', 9: 'abcdfg'}
        # list of list data structure for storing the number of wires in common between digits
        self.inCommon = [[len(set(self.baseMap[i])&set(self.baseMap[j])) for i in range(10)] for j in range(10)]
        data = self.loadData()
        self.outputs = [x[x.find('|')+2:].split() for x in data]
        self.inputs = [x[:x.find('|')].split() for x in data]
        self.vis = set()
        self.patterns = list()

    def loadData(self, path = "inputs/input.txt"):
        with open(path, "r") as f:
            A = f.read().splitlines()
        return A
    
    def getOutput(self, i):
        self.patterns = ['$']*10
        self.vis = set()
        self.dfs(i, 0)
        patToDig = self.setCharDigits()
        outputDigits = 0
        for word in self.outputs[i]:
            sword = "".join(sorted(word)) # sort of the word
            outputDigits = (outputDigits*10) + patToDig[sword]
        return outputDigits

    def computeSumOutputs(self):
        return sum(self.getOutput(i) for i in range(len(self.inputs)))
            
    def setCharDigits(self):
        patToDig = dict()
        for dig, pat in enumerate(self.patterns):
            patToDig[pat] = dig
        return patToDig

    def dfs(self, index, jindex):
        """
        index: index of the inputs, it is a single line of 10 character strings
        jindex: the index of the 10 character string in the current inputs[index]
        """
        if jindex == 10:
            for i in range(10):
                for j in range(i+1, 10):
                    if len(set(self.patterns[i])&set(self.patterns[j])) != self.inCommon[i][j]:
                        return False
            return True
        word = "".join(sorted(self.inputs[index][jindex]))
        if word in self.vis:
            return False
        for digit in range(10):
            charLength = len(self.baseMap[digit])
            if charLength == len(word) and self.patterns[digit]=='$' and word not in self.vis:
                self.patterns[digit] = word
                self.vis.add(word)
                if self.dfs(index, jindex + 1):
                    return True
                self.patterns[digit] = '$'
                self.vis.remove(word)
        return False

if __name__ == "__main__":
    s = SevenSegmentSearch()
    print(s.computeSumOutputs())
```

Uses logic to solve with if statements. Using frozenset because it is immutable set that is hashable and can be a key in a dictionary in python. 

This was another way to solve.  It turned out that you can separate the 2,3,5 and 0,6,9 into 
sets that have 5 and 6 segments respectively. Then there exist relationships that would
indicate the digit.  For example if length if 5 and it has 2 segments in common with the 1 digit. 
It must be a 3.  

```py
class SevenSegmentSearch:
    def dataLoader(self, path = "inputs/input.txt"):
        with open("inputs/input.txt", "r") as f:
            loadedData = f.read().splitlines()
        for data in loadedData:
            rawPatterns, rawDigits = map(str.split, data.split('|'))
            patterns = tuple(map(lambda x: (frozenset(x), len(x)), rawPatterns))
            digits = tuple(map(lambda x: frozenset(x), rawDigits))
            yield patterns, digits

    def deduceMapping(self, patterns):
        # pattern to digit mapping
        p2d = dict()
        for p, plen in patterns:
            if plen == 2:
                p2d[p] = 1
            elif plen == 3:
                p2d[p] = 7
            elif plen == 4:
                p2d[p] = 4
            elif plen == 7:
                p2d[p] = 8
        d2p = {v: k for k, v in p2d.items()}
        for p, plen in patterns:
            # 3 or 5 or 2, all have length of 5
            if plen==5:
                if len(p & d2p[1]) == 2:
                    p2d[p] = 3
                elif len(p&d2p[4]) == 3:
                    p2d[p] = 5
                else:
                    p2d[p] = 2
            elif plen==6:
                if len(p&d2p[4])==4:
                    p2d[p] = 9
                elif len(p&d2p[7])==2:
                    p2d[p] = 6
                else:
                    p2d[p] = 0
        return p2d

    def getOutput(self):
        totalCnt = 0
        for pattern, digit in self.dataLoader():
            p2d = self.deduceMapping(pattern)
            cnt = 0
            for dig in digit:
                cnt = cnt*10 + p2d[dig]
            totalCnt += cnt
        return totalCnt
```

## Day 9

### Part 1

We are given 100 digits in 100 lines
To parse the data I convert from the file to a string with f.read(), then I use split to 
create a list that uses whitespace delimiter by default so it will use the line break. 
And then I can iterate through these strings to create and array of integers.

I used a sum function the value + 1 for the lowest points by checking that all the locations around it are larger if
it is within the grid. 

```py
data = []
lines = f.read().split()
for line in lines:
    data.append([int(x) for x in line])
R, C = len(data), len(data[0])
sumRisk = sum(data[i][j] + 1 for i in range(R) for j in range(C) if all(data[i][j] < data[i+dr][j+dc] for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)) if 0 <= i+dr < R and 0 <= j+dc < C))
print(sumRisk)
```

### Part 2

This part is trickier, but it turns out you can use the fact that 9s are never going to be part of a basin as stipulated in 
puzzle statement. 

Let's look at a particularly interesting edge case

999
979
289

This will return two basins [2,2] of size 2, but that can't be right, cause both basins have the 8, but is that possible.
It says in the statement a basins is all locations that eventually flow downward to a single low point.  But 8 is flowing
down to two low points. And it states all location will be a part of exactly one basin.  So how can 8 be a part of 2 basins.  
So we can't just do a simple bfs that moves outward to neighbors that are larger. 

This following dfs solution use memoization to avoid it computing 8 as being part of two basins. 
So basically it is dfs so it goes towards a path that leads to a low point.  
Then it will return that low point.  It say I hit a point that I already know the low point is, it will just return
that via the memoization.  It is most efficient to save values from previous.  


```py
data = []
lines = f.read().split()
for line in lines:
    data.append([int(x) for x in line])
R, C = len(data), len(data[0])
@lru_cache(maxsize=None)
def dfs(x, y):
    for i, j in ((x, y-1), (x, y+1), (x-1, y), (x+1, y)):
        if 0 <= i < R and 0 <= j < C and data[i][j] < data[x][y]:
            return dfs(i,j)
    return (x,y)
basins = Counter(dfs(i,j) for i in range(R) for j in range(C) if data[i][j] != 9)
heights = sorted(list(basins.values()))
print(heights[-1]*heights[-2]*heights[-3])
```

## Day 10

### Part 1

Using a stack and dictionary to find the first error and increase the error score by the points for each
syntax breaking bracket.  

```py
class SyntaxScoring:
    def __init__(self):
        self.errorScore = 0
        self.openings = {'{': '}', '(': ')', '[': ']', '<': '>'}
        self.points = {k: v for k, v in zip((')', ']', '}', '>'), (3, 57, 1197, 25137))}
    def isCorrupted(self, chunk):
        stk = []
        for c in chunk:
            if c in self.openings:
                stk.append(c)
            else:
                prev = stk.pop()
                if self.openings[prev] != c:
                    self.errorScore += self.points[c]
                    return True
        return False
    def dataLoader(self):
        with open("inputs/input.txt", "r") as f:
            return f.read().splitlines()
    def run(self):
        data = self.dataLoader()
        for chunk in data:
            self.isCorrupted(chunk)
        return self.errorScore
if __name__ == '__main__':
    print(SyntaxScoring().run())
```

### Part 2

Stack solution as well but create a list of 


```py
from functools import reduce
class SyntaxScoring:
    def __init__(self):
        self.openings = {'{': '}', '(': ')', '[': ']', '<': '>'}
        self.points = {k: v for k, v in zip((')', ']', '}', '>'), range(1,5))}
        self.closers = []
    def isCorrupted(self, chunk):
        """
        Returns true if it is corrupted
        if it is not corrupted it will add the sequence of closing characters
        to a global variable self.closings to make the line complete. 
        """
        stk = []
        for c in chunk:
            if c in self.openings:
                stk.append(c)
            else:
                prev = stk.pop()
                if self.openings[prev] != c:
                    return True
        self.closers = [self.openings[c] for c in stk[::-1]]
        return False
    def dataLoader(self):
        with open("inputs/input.txt", "r") as f:
            return f.read().splitlines()

    def run(self):
        """
        Returns the middle value in the scores for the necessary sequence of closing characters for
        the incomplete lines.
        """
        data = self.dataLoader()
        scores = [reduce(lambda a, b: a*5 + b, (self.points[c] for c in self.closers), 0) for chunk in data if not self.isCorrupted(chunk)]
        scores.sort()
        return scores[len(scores)//2]
if __name__ == '__main__':
    print(SyntaxScoring().run())
```

## Day 11

### Part 2: 

```cpp

```

## Day 12

### Part 2: 

DFS + backtracking to find all the paths but we have some extra caveats such as 
we can visit lower case letters only once, but uppercase we can visit infinitely many times
so I don't add uppercase to the visited set.  I only add lowercase to the visited set.


```py
from collections import defaultdict
with open("inputs/input.txt", "r") as f:
    raw_data = [(x,y) for x,y in line.split('-') for line in f.read().split('\n')]
    print(raw_data)
    graph = defaultdict(list)
    for x, y in raw_data:
        graph[x].append(y)
        graph[y].append(x)
    visited = {'start'}
    def dfs(parent, node):
        if node == 'end':
            return 1
        paths = 0
        for nei in graph[node]:
            if (parent.islower() and nei==parent) or nei in visited:
                continue
            if nei.islower():
                visited.add(nei)
            paths += dfs(node, nei)
            if nei.islower():
                visited.remove(nei)
        return paths
    print(dfs('none', 'start'))
```

## Day 13


### Part 1

I create tuples of the instructions and have a function for folding along x and y axis. 


```py
class TransparentOrigami:
    def __init__(self):
        self.data = set()
        self.maxY = 0
        self.maxX = 0
        self.folds = []
    def dataLoader(self):
        with open("inputs/input.txt", "r") as f:
            points, folds = f.read().split('\n\n')
            self.data = {tuple(map(int,points.split(','))) for points in points.split('\n')}
            self.folds = [(fold[11], int(fold[13:])) for fold in folds.split('\n')]
    def fold(self, axis, n):
        if axis == 'x':
            return {(2*n-x, y) if x > n else (x,y) for x,y in self.data}
        return {(x, 2*n-y) if y > n else (x,y) for x,y in self.data}
    def run(self):
        self.dataLoader()
        for axis, n in self.folds:
            self.data = self.fold(axis,n)
            return len(self.data) # just for part 1 cause it needs to just do one fold
        return len(self.data)

if __name__ == '__main__':
    s = TransparentOrigami()
    print(s.run())
```

### Part 2

```py
class TransparentOrigami:
    def __init__(self):
        self.data = set()
        self.maxY = 0
        self.maxX = 0
        self.folds = []
    def dataLoader(self):
        with open("inputs/input.txt", "r") as f:
            points, folds = f.read().split('\n\n')
            self.data = {tuple(map(int,points.split(','))) for points in points.split('\n')}
            self.folds = [(fold[11], int(fold[13:])) for fold in folds.split('\n')]
    def fold(self, axis, n):
        if axis == 'x':
            return {(2*n-x, y) if x > n else (x,y) for x,y in self.data}
        return {(x, 2*n-y) if y > n else (x,y) for x,y in self.data}
    def run(self):
        self.dataLoader()
        for axis, n in self.folds:
            self.data = self.fold(axis,n)
        return self.displayData()
    def displayData(self):
        self.maxX = 0
        self.maxY = 0
        for x, y in self.data:
            self.maxX = max(self.maxX, x)
            self.maxY = max(self.maxY, y)
        grid = [[' ' for x in range(self.maxX+1)] for y in range(self.maxY+1)]
        for x, y in self.data:
            grid[y][x] = '#'
        return "\n".join(["".join(row) for row in grid])

if __name__ == '__main__':
    s = TransparentOrigami()
    print(s.run())
```

## Day 14

### Part 2: 

I store offline all the possible transitions from pair of characters to the two pair of characters it creates.  
I store this in a dictionary called transitions.  Then I create a Counter called freqs for storying the count of each pair of characters from template.  

I create a third Counter to store the counts of each character.  

Then all we have to do is iterate through the 100 possible transitions to create the next freqs and increase the count for the inserted character.  

```py
from collections import Counter
with open("inputs/input.txt", "r") as f:
    freqs = Counter()
    template = f.readline().strip()
    f.readline()
    raw_data = f.read().splitlines()
    for k, v in zip(template, template[1:]):
        s = k + v
        freqs[s]+=1
    transitions = {}
    for line in raw_data:
        x, y = line.split(" -> ")
        transitions[x] = [x[0]+y, y+x[1]]
    counts = Counter(template)
    for _ in range(40):
        tmp = Counter()
        for pair, count in freqs.items():
            if pair in transitions:
                counts[transitions[pair][0][1]] += count
                for new_pair in transitions[pair]:
                    tmp[new_pair] += count
        freqs = tmp
    mostCommon = counts.most_common()
    print(mostCommon[0][1]-mostCommon[-1][1])
```

## Day 15

### Part 1

Dijkstra algorithm with a visited set, to get the cheapest path through the grid from top left to bottom right.

Using a minheap datastructure

```py
import heapq
with open("inputs/input.txt", "r") as f:
    data = []
    lines = f.read().splitlines()
    for line in lines:
        data.append([int(x) for x in line])
    heap = []
    R, C = len(data), len(data[0])
    heapq.heappush(heap,(0,0,0))
    vis = set()
    vis.add((0,0))
    while heap:
        cost, r, c = heapq.heappop(heap)
        if r==R-1 and c==C-1:
            print(cost)
            break
        for dr in range(-1,2):
            for dc in range(-1,2):
                if abs(dr+dc)==1:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<R and 0<=nc<C and (nr,nc) not in vis:
                        vis.add((nr,nc))
                        heapq.heappush(heap,(cost+data[nr][nc],nr,nc))
```

### Part 2

Really the only caveat is that I need to add to the original grid and still return the min cost, so still dijkstra but with 
modification


```py
import heapq
with open("inputs/input.txt", "r") as f:
    data = []
    lines = f.read().splitlines()
    for line in lines:
        data.append([int(x) for x in line])
    heap = []
    R, C = len(data), len(data[0])
    for k in range(1,5):
        for j in range(C):
            for i in range((k-1)*R,k*R):
                if i+R == len(data):
                    data.append([])
                data[i+R].append(data[i][j]+1 if data[i][j]<9 else 1)
    R = len(data)
    for k in range(1,5):
        for i in range(R):
            for j in range((k-1)*C, k*C):
                data[i].append(data[i][j]+1 if data[i][j]<9 else 1)
    C = len(data[0])
    heapq.heappush(heap,(0,0,0))
    vis = set()
    vis.add((0,0))
    while heap:
        cost, r, c = heapq.heappop(heap)
        if r==R-1 and c==C-1:
            print(cost)
            break
        for dr in range(-1,2):
            for dc in range(-1,2):
                if abs(dr+dc)==1:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<R and 0<=nc<C and (nr,nc) not in vis:
                        vis.add((nr,nc))
                        heapq.heappush(heap,(cost+data[nr][nc],nr,nc))
```

Improvement on this would be to do online generation of the map fromt he single grid we start with.  Then I just need to generate the 
values from the original with a formula and that would make it more efficient. 


Let's do the math

We have a grid with R rows and C cols

I begin by taking values from within, but eventually
nr >= R suppose, in which case nr/R = 1
So to get it's value I need to take nr - 1*R from the original grid, which takes me to row=0, then I need to add to the value data[nr][nc] + 1
but if data[nr][nc]+1>9 then I need to actually add data[nr][nc]+1-9.
Sure and the same for nc>=C

Consider the next case:  nr>=R and nc>=C 

You can get it's value will be if x=nr/R and y = nc/C
then we have data[nr][nc] + x + y

Solves it in 1.2 seconds on my hardware.  

```py
import heapq
with open("inputs/input.txt", "r") as f:
    data = []
    lines = f.read().splitlines()
    for line in lines:
        data.append([int(x) for x in line])
    heap = []
    R, C = len(data), len(data[0])
    heapq.heappush(heap,(0,0,0))
    vis = set()
    vis.add((0,0))
    while heap:
        cost, r, c = heapq.heappop(heap)
        if r==5*R-1 and c==5*C-1:
            print(cost)
            break
        for dr in range(-1,2):
            for dc in range(-1,2):
                if abs(dr+dc)==1:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<5*R and 0<=nc<5*C and (nr,nc) not in vis:
                        x, y = nr//R, nc//C
                        nval = data[nr-x*R][nc-y*C] + x + y
                        nval = nval if nval<=9 else nval-9
                        vis.add((nr,nc))
                        heapq.heappush(heap,(cost+nval,nr,nc))
```

## Day 16

### Part 2: 

Just parsing the packets and breaking it down into 3 parts,
parsing the literal, parsing the operator, and parsing in general

parsing in general is for when you don't know if it is a literal or operator. 

```py
class PacketDecoder:
    def __init__(self, hexa):
        self.version = 0
        self.i = 0
        self.binary_data = self.conv_hexa_binary(self.data_loader())
    
    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            return f.read()

    def conv_hexa_binary(self, hexa):
        hexBin = {'0': '0000', '1': '0001', '2': '0010', '3': '0011', '4': '0100', '5': '0101', '6': '0110', 
        '7': '0111', '8': '1000', '9': '1001', 'A': '1010', 'B': '1011', 'C': '1100', 'D': '1101', 'E': '1110', 'F': '1111'}
        binary = ""
        for c in hexa:
            binary += hexBin[c]
        return binary

    def get(self, size):
        val = int(self.binary_data[self.i:self.i+size],2)
        self.i += size
        return val
    
    def get_binary(self, size):
        b = self.binary_data[self.i:self.i+size]
        self.i+=size
        return b

    def parse_literal(self):
        binary = ""
        while True:
            A = self.get_binary(5)
            binary += A[1:]
            if A[0] == '0':
                break
        return int(binary, 2)

    def parse_operator(self, typ):
        initialMap = {0: 0, 1:1, 2: 10000, 3:-10000, 5: [], 6: [], 7:[]}
        initial = initialMap[typ]
        len_id = self.get(1)
        if len_id==0:
            num_bits = self.get(15)
            starting_bit = self.i
            while self.i-starting_bit<num_bits:
                val = self.parser()
                if typ==0:
                    initial += val
                elif typ==1:
                    initial*=val
                elif typ==2:
                    initial = min(initial, val)
                elif typ==3:
                    initial = max(initial, val)
                else:
                    initial.append(val)
        else:
            num_subpackets = self.get(11)
            for _ in range(num_subpackets):
                val = self.parser()
                if typ==0:
                    initial += val
                elif typ==1:
                    initial*=val
                elif typ==2:
                    initial = min(initial, val)
                elif typ==3:
                    initial = max(initial, val)
                else:
                    initial.append(val)
        if typ==5:
            return 1 if initial[0]>initial[1] else 0
        elif typ==6:
            return 1 if initial[0]<initial[1] else 0
        elif typ==7:
            return 1 if initial[0]==initial[1] else 0
        return initial
    def parser(self):
        self.version += self.get(3)
        pid = self.get(3)
        if pid==4:
            return self.parse_literal()
        return self.parse_operator(pid)
    def run(self):
        return self.parser()
    
if __name__ == '__main__':
    s = PacketDecoder(None)
    print(f'the sum of the packet versions: {s.version}') # part 1
    print(f'the result of parsing the hexadecimal: {s.run()}') # part 2
```

discord user solution

Going to analyze this, there are a lot of tricks I think are so beautiful
```py
from math import *
from operator import *
s = ''.join(bin(int(c, 16))[2:].zfill(4) for c in input().strip())
ans = pos = 0
def get(a):
    global pos
    return int(s[pos:(pos:=pos+a)], 2)
def fun(a, b):
    return a*b
def parse():
    global ans
    version = get(3)
    ans += version
    typeid = get(3)
    if typeid == 4:
        l = ''
        while s[pos] == '1':
            l += bin(get(5)%16)[2:].zfill(4)
        l += bin(get(5)%16)[2:].zfill(4)
        return int(l, 2)
    l = []
    if get(1):
        numpackets = get(11)
        for p in range(numpackets):
            l.append(parse())
    else:
        z = get(15)+pos
        while pos < z:
            l.append(parse())
    return [sum, prod, min, max][typeid](l) if typeid < 4 else [gt, lt, eq][typeid-5](*l)
print('Part 2:', parse())
print('Part 1:', ans)

```

## Day 17

### Part 2: 

```py
import re
with open("inputs/input.txt", "r") as f:
    xmin, xmax, ymin, ymax = list(map(int, re.findall(r'[-\d]+', f.read())))
    cnt = 0
    print(abs(ymin)*abs(ymin+1)//2) # part 1
    for dx_init in range(min(0, xmin-1), max(0, xmax) + 1):
        for dy_init in range(ymin,abs(ymin)+1):
            dx,dy=dx_init,dy_init
            x, y = 0, 0
            while y>ymin:
                y+=dy
                x+=dx
                if dx>0:
                    dx-=1
                if dx<0:
                    dx+=1
                dy-=1
                if xmin<=x<=xmax and ymin<=y<=ymax:
                    cnt+=1
                    break
    print(cnt) # part 2
        
```

## Day 18

### Part 2: 

```cpp

```

## Day 19

### Part 2: 

```py
import numpy as np
from collections import defaultdict, deque, Counter
class BeaconScanner:
    def __init__(self):
        self.data = None
        self.delta = None # The relative position between two scanners

    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            self.data = [[np.fromiter(map(int, coords.split(',')), dtype=int) for coords in lst.split('\n')[1:]] for lst in f.read().split('\n\n')]
    """
    TODO: Write this in a numpy method that generates the rotation matrices without hard coding
    Quick solution is to hard code the rotation matrices
    """
    def rotations(self):
        yield np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=int)
        yield np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=int)
        yield np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=int)
        yield np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=int)
        yield np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=int)
        yield np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=int)
        yield np.array([[0,1,0],[1,0,0],[0,0,-1]], dtype=int)
        yield np.array([[0,0,-1],[1,0,0],[0,-1,0]], dtype=int)
        yield np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=int)
        yield np.array([[-1,0,0],[0,0,-1],[0,-1,0]], dtype=int)
        yield np.array([[-1,0,0],[0,1,0],[0,0,-1]], dtype=int)
        yield np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=int)
        yield np.array([[0,1,0],[-1,0,0],[0,0,1]], dtype=int)
        yield np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=int)
        yield np.array([[0,-1,0],[-1,0,0],[0,0,-1]], dtype=int)
        yield np.array([[0,0,-1],[-1,0,0],[0,1,0]], dtype=int)
        yield np.array([[0,0,-1],[0,1,0],[1,0,0]], dtype=int)
        yield np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=int)
        yield np.array([[0,0,1],[0,-1,0],[1,0,0]], dtype=int)
        yield np.array([[0,-1,0],[0,0,-1],[1,0,0]], dtype=int)
        yield np.array([[0,0,-1],[0,-1,0],[-1,0,0]], dtype=int)
        yield np.array([[0,-1,0],[0,0,1],[-1,0,0]], dtype=int)
        yield np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=int)
        yield np.array([[0,1,0],[0,0,-1],[-1,0,0]], dtype=int)

    def rotation_beacons(self, scanner):
        for rotation in self.rotations():
            beacons = []
            for beacon in self.data[scanner]:
                beacons.append(np.matmul(rotation, beacon))
            yield beacons
            
    def matches(self, scan1, scan2, rot, initial_beacon1, initial_beacon2, dx, dy, dz):
        # create a set of all the beacons in scan1
        seen = set(map(tuple, self.data[scan1]))
        numMatches = 1
        for i, beacon in enumerate(self.data[scan2]):
            if i == initial_beacon2:
                continue
            beacon = np.matmul(rot, beacon)
            cand = (beacon[0]-dx, beacon[1]-dy, beacon[2]-dz)
            if cand in seen:
                numMatches += 1
        return numMatches

    def findsMatch(self, located_scanner, unlocated_scanner):
        for beacons in self.rotation_beacons(unlocated_scanner):
            count = Counter()
            for beacon2 in self.data[located_scanner]:
                for beacon in beacons:
                    new_beacon = beacon-beacon2
                    count[tuple(new_beacon)] += 1
                    for k, cnt in count.items():
                        if cnt==12:
                            self.data[unlocated_scanner] = set()
                            for beacon in beacons:
                                self.data[unlocated_scanner].add(tuple(beacon-beacon2))
                            return True
        return False

    def run(self):
        self.data_loader()
        numScanners = len(self.data)
        located_scanners = set([0])
        unlocated_scanners = set(range(1,numScanners))
        while len(unlocated_scanners)>0:
            print(len(unlocated_scanners))
            for i in range(numScanners):
                if i in located_scanners:
                    continue
                for j in located_scanners:
                    if self.findsMatch(j,i):
                        located_scanners.add(i)
                        unlocated_scanners.remove(i)
                        break
        # add up all the beacons that are measured relative to scanner 0 from all the other scanners
        # This will be the total count of beacons in the water
        res = set()
        for i in range(numScanners):
            for beacon in self.data[i]:
                res.add(tuple(beacon))
        print(len(res))
                    




if __name__ == '__main__':
    bs = BeaconScanner()
    bs.run()

```

## Day 20

### Part 2: 


The solution to part 2, but it really works for part 1 as well just with 2 instead of 50 iterations.  

Basic idea is to understand the trick in the input, sense the self.algorithms[0]=#, that means all of the '.' in the infinite
input image will be switched to '#' but then switch to '.' again because self.algorithms[-1]='.'.  So we just need to ignore all the outer
ones and instead just consider a padding of -2 and +2 for the rows and columns, because those are the only ones that we do not know yet.
Also all the outside values will cancel out.  


if You draw a picture 


```
.......
.......
.......
...#...
....#..
```

which points do you need to compute? Which points are relevant? 

```py
INF = 100000
class imageEnhancement:
    def __init__(self):
        self.algorithm = None
        self.data = None

    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            lst = f.read().split('\n')
            self.algorithm = lst[0]
            data = lst[2:]
            self.data = set()
            for i in range(len(data)):
                for j in range(len(data[0])):
                    if data[i][j] == '#':
                        self.data.add((i, j))

    def enhance(self, iteration):
        row_min, row_max, col_min, col_max = INF, -INF, INF, -INF
        rows, cols = zip(*self.data)
        row_min, row_max, col_min, col_max = min(rows), max(rows), min(cols), max(cols)
        new_lights = set()
        def improve(row, col):
            binary_value = ""
            for i in range(row - 1, row + 2):
                for j in range(col - 1, col + 2):
                    if i<row_min or i>row_max or j<col_min or j>col_max:
                        binary_value += str(iteration)
                        continue
                    binary_value += '1' if (i,j) in self.data else '0'
            i = int(binary_value, 2)
            if self.algorithm[i] == '#':
                new_lights.add((row, col))
        for i in range(row_min-2, row_max+3):
            for j in range(col_min-2,col_max+3):
                improve(i, j)
        self.data = new_lights
    def run(self):
        self.data_loader()
        for i in range(50):
            self.enhance(i%2)
        print(len(self.data))

if __name__ == "__main__":
    imageEnhancement().run()
```

## Day 21

### Part 1

Solution: Simulate the game via iteration

```py
def run(self):
    pos1, pos2= map(int, self.data_loader())
    dice, score1, score2, turn, cnt = -1, 0, 0, 0, 0
    while score1<1000 and score2<1000:
        cnt+=1
        if turn==0:
            moves = 0
            for _ in range(3):
                dice = (dice+1)%100
                moves += dice +1
            pos1 = (pos1+moves)%10
            score1 += pos1+1
        else:
            moves = 0
            for _ in range(3):
                dice = (dice+1)%100
                moves += dice +1
            pos2 = (pos2+moves)%10
            score2 += pos2+1     
        turn^=1
    return min(score1,score2)*cnt*3
```

### Part 2

Solution: Iterative DP with states

```py
class DiracDice:
    def __init__(self):
        self.states = [[[[[0 for _ in range(2)] for _ in range(10)] for _ in range(10)] for _ in range(21)] for _ in range(21)]
        # states[score1][score2][p1][p2][turn] = the number of universes with this state
        self.wins1, self.wins2 = 0, 0
    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            data = f.read().split("\n")
            return int(data[0].split(': ')[1])-1, int(data[1].split(': ')[1])-1
    def run(self):
        initial_pos1, initial_pos2 = self.data_loader()
        print(initial_pos1,initial_pos2)
        self.states[0][0][initial_pos1][initial_pos2][0] = 1
        curStates = set([(0,0,initial_pos1,initial_pos2,0)])
        while curStates:
            newStates = set()
            for score1, score2, pos1, pos2, turn in curStates:
                cnt = self.states[score1][score2][pos1][pos2][turn]
                self.states[score1][score2][pos1][pos2][turn] = 0
                if turn==0:
                    for i in range(1,4):
                        for j in range(1,4):
                            for k in range(1,4):
                                pos = (pos1+i+j+k)%10
                                score = score1+pos+1
                                if score>=21:
                                    self.wins1+=cnt
                                else:
                                    self.states[score][score2][pos][pos2][turn^1] += cnt
                                    newStates.add((score,score2,pos,pos2,turn^1))
                else:
                    for i in range(1,4):
                        for j in range(1,4):
                            for k in range(1,4):
                                pos = (pos2+i+j+k)%10
                                score = score2+pos+1
                                if score>=21:
                                    self.wins2+=cnt
                                else:
                                    self.states[score1][score][pos1][pos][turn^1] += cnt
                                    newStates.add((score1,score,pos1,pos,turn^1))
            curStates = newStates
        return max(self.wins1,self.wins2)
if __name__ == '__main__':
    dd = DiracDice()
    print(dd.run())
```

Solution: Recursive DP with states and memoization with cache


```py
from functools import cache
class DiracDice:
    def __init__(self):
        self.states = [[[[[0 for _ in range(2)] for _ in range(10)] for _ in range(10)] for _ in range(21)] for _ in range(21)]
        # states[score1][score2][p1][p2][turn] = the number of universes with this state
        self.wins1, self.wins2 = 0, 0
    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            data = f.read().split("\n")
            return int(data[0].split(': ')[1])-1, int(data[1].split(': ')[1])-1
    @cache
    def simulateUniverses(self, score1, score2, pos1, pos2, turn):
        wins1, wins2 = 0, 0
        if turn==0:
            for i in range(1,4):
                for j in range(1,4):
                    for k in range(1,4):
                        pos = (pos1+i+j+k)%10
                        score = score1+pos+1
                        if score>=21:
                            wins1 +=1
                        else:
                            nwins1, nwins2 =  self.simulateUniverses(score, score2, pos, pos2, turn^1)
                            wins1 += nwins1
                            wins2 += nwins2
        else:
            for i in range(1,4):
                for j in range(1,4):
                    for k in range(1,4):
                        pos = (pos2+i+j+k)%10
                        score = score2+pos+1
                        if score>=21:
                            wins2 +=1
                        else:
                            nwins1, nwins2 =  self.simulateUniverses(score1, score, pos1, pos, turn^1)
                            wins1 += nwins1
                            wins2 += nwins2
        return wins1, wins2
    def run(self):
        initial_pos1, initial_pos2 = map(int, self.data_loader())
        return max(self.simulateUniverses(0, 0, initial_pos1, initial_pos2, 0))
if __name__ == '__main__':
    dd = DiracDice()
    print(dd.run())
```


Solution: Recursive DP but reduced the arguments

Use the fact that you switch players each turn, so you are computing the other wins and other losses, and the opponent losses count towards your wins.
And you switch who will be the current player in the next recursive call.  This way I don't need boolean for storing the turn.  

```py
@cache
def simulateUniverses(self, cur_score, other_score, cur_pos, other_pos):
    if cur_score>=21 or other_score>=21:
        return cur_score>=21, other_score>=21
    wins, losses = 0, 0
    for i in range(1,4):
        for j in range(1,4):
            for k in range(1,4):
                pos = (cur_pos+i+j+k)%10
                score = cur_score +pos+1
                other_wins,other_losses = self.simulateUniverses(other_score, score, other_pos, pos)
                wins += other_losses
                losses += other_wins
    return wins, losses
```

## Day 22

### Part 2 

Part 1 is the same but it limits the input so you can get away with a more brute forcy solution.  

For part 2 you have such a large input you need to keep track of the cuboids appropriately.  

Basic idea is to store the cuboids in a counter, and keep count.  So when you turn a cuboid on, you turn on that cuboid in the counter
But you need to make sure that if there are any overlaps with previous cuboids that you remove those because when you add your current cuboid
it will be double counting the cells that overlap with other cuboids. 

Then if you are turning off you just want to remove the overlapping with other cuboids
```py
import re
from collections import namedtuple, Counter
class Instruction:
    def __init__(self, status, coords):
        self.status = status
        Range = namedtuple("Range", ['initial', 'final'])
        self.xrange = Range(coords[0], coords[1])
        self.yrange = Range(coords[2], coords[3])
        self.zrange = Range(coords[4], coords[5])
    def __repr__(self):
        return f"command: {self.status}, x = ({self.xrange.initial}, {self.xrange.final}), y = ({self.yrange.initial}, {self.yrange.final}, z = ({self.zrange.initial}, {self.zrange.final})"
class ReactorReboot:
    def __init__(self):
        self.instructions = None

    def data_loader(self):
        with open("inputs/input.txt","r") as f:
            return [Instruction(line.split()[0], list(map(int, re.findall(r'[-\d]+', line)))) for line in f.read().splitlines()]

    def run(self):
        self.instructions = self.data_loader()
        cuboids = Counter()
        for instruction in self.instructions:
            new_cuboids = Counter()
            nx0, nx1, ny0, ny1, nz0, nz1 = instruction.xrange.initial,instruction.xrange.final, instruction.yrange.initial, instruction.yrange.final, instruction.zrange.initial, instruction.zrange.final
            for (x0,x1,y0,y1,z0,z1), sgn in cuboids.items():
                bx0, bx1, by0, by1, bz0, bz1 = max(x0,nx0), min(x1,nx1), max(y0,ny0), min(y1,ny1), max(z0,nz0), min(z1,nz1)
                if bx0<=bx1 and by0<=by1 and bz0<=bz1:
                    new_cuboids[(bx0,bx1,by0,by1,bz0,bz1)] -= sgn
            if instruction.status=="on":
                new_cuboids[(nx0,nx1,ny0,ny1,nz0,nz1)] += 1
            cuboids.update(new_cuboids)
        return sum((x1-x0+1)*(y1-y0+1)*(z1-z0+1)*sgn for (x0,x1,y0,y1,z0,z1), sgn in cuboids.items())

if __name__ == '__main__':
    reactor = ReactorReboot()
    print(reactor.run())
```

## Day 23

### Part 2: 

```py
import collections
import itertools
import math
import re

goal = {
    'A': 2,
    'B': 4,
    'C': 6,
    'D': 8,
}
goalSpaces = set(goal.values())
moveCosts = {
    'A': 1,
    'B': 10,
    'C': 100,
    'D': 1000,
}


def canReach(board, pos, dest):
    a = min(pos, dest)
    b = max(pos, dest)
    for i in range(a, b+1):
        if i == pos:
            continue
        if i in goalSpaces:
            continue
        if board[i] != '.':
            # print(' ', i, board[i][0], 'cannot reach')
            return False
    return True


def roomOnlyContainsGoal(board, piece, dest):
    inRoom = board[dest]
    return len(inRoom) == inRoom.count('.') + inRoom.count(piece) 


def getPieceFromRoom(room):
    for c in room:
        if c != '.':
            return c


def possibleMoves(board, pos):
    piece = board[pos]
    # print(board, pos, piece)
    if pos not in goalSpaces:
        if canReach(board, pos, goal[piece]) and roomOnlyContainsGoal(board, piece, goal[piece]):
            return [goal[piece]]
        return []

    movingLetter = getPieceFromRoom(piece)
    if pos == goal[movingLetter] and roomOnlyContainsGoal(board, movingLetter, pos):
        return []

    possible = []
    for dest in range(len(board)):
        if dest == pos:
            continue
        if dest in goalSpaces and goal[movingLetter] != dest:
            continue
        if goal[movingLetter] == dest:
            if not roomOnlyContainsGoal(board, movingLetter, dest):
                continue
        if canReach(board, pos, dest):
            possible.append(dest)
    return possible


def addToRoom(letter, room):
    room = list(room)
    dist = room.count('.')
    assert dist != 0
    room[dist-1] = letter
    return ''.join(room), dist


def move(board, pos, dest):
    new_board = board[:]
    dist = 0
    movingLetter = getPieceFromRoom(board[pos])
    if len(board[pos]) == 1:
        new_board[pos] = '.'
    else:
        new_room = ''
        found = False
        for c in board[pos]:
            if c == '.':
                dist += 1
                new_room += c
            elif not found:
                new_room += '.'
                dist += 1
                found = True
            else:
                new_room += c
        new_board[pos] = new_room
    
    dist += abs(pos - dest)

    if len(board[dest]) == 1:
        new_board[dest] = movingLetter
        return new_board, dist * moveCosts[movingLetter]
    else:
        new_board[dest], addl_dist = addToRoom(movingLetter, board[dest])
        dist += addl_dist
        return new_board, dist * moveCosts[movingLetter]


def solve(board):
    states = {tuple(board): 0}
    queue = [board]
    while queue:
        # print(len(queue))
        board = queue.pop()
        for pos, piece in enumerate(board):
            if getPieceFromRoom(piece) is None:
                continue
            dests = possibleMoves(board, pos)
            # print('{} ({}) can move to {}'.format(piece, pos, dests))
            for dest in dests:
                new_board, addl_cost = move(board, pos, dest)
                new_cost = states[tuple(board)] + addl_cost
                new_board_tuple = tuple(new_board)
                cost = states.get(new_board_tuple, 9999999)
                if new_cost < cost:
                    # print(board, '->', new_board, ':', new_cost)
                    states[new_board_tuple] = new_cost
                    queue.append(new_board)

    return states

board = ['.', '.', 'AB', '.', 'DC', '.', 'BA', '.', 'DC', '.', '.']
states = solve(board)  # 12240
print(states[('.', '.', 'AA', '.', 'BB', '.', 'CC', '.', 'DD', '.', '.')])

board = ['.', '.', 'ADDB', '.', 'DCBC', '.', 'BBAA', '.', 'DACC', '.', '.']
states = solve(board)  # 44618
print(states[('.', '.', 'AAAA', '.', 'BBBB', '.', 'CCCC', '.', 'DDDD', '.', '.')])
```

## Day 24

### Part 2: 

```cpp

```

## Day 25


### Part 1

Solution:  Brute force algorithm to move the sea cucumbers in the grid.  

```py
class SeaCucumber:
    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            return [list(line) for line in f.read().split('\n')]
    def run(self):
        data = self.data_loader()
        steps, R, C = 0, len(data), len(data[0])
        while True:
            steps += 1
            ndata = [line[:] for line in data]
            found = False
            for i in range(R):
                for j in range(C):
                    if ndata[i][j]=='>' and ndata[i][(j+1)%C]=='.':
                        ndata[i][j]='r'
                        ndata[i][(j+1)%C]='x'
                        found = True
            for i in range(R):
                for j in range(C):
                    if ndata[i][j]=='r':
                        ndata[i][j]='.'
                    if ndata[i][j]=='x':
                        ndata[i][j]='>'
            for j in range(C):
                for i in range(R):
                    if ndata[i][j]=='v' and ndata[(i+1)%R][j]=='.':
                        ndata[i][j]='r'
                        ndata[(i+1)%R][j]='y'
                        found = True
            for i in range(R):
                for j in range(C):
                    if ndata[i][j]=='r':
                        ndata[i][j]='.'
                    if ndata[i][j]=='y':
                        ndata[i][j]='v'
            if not found:
                break
            data = ndata
        print(steps)
if __name__ == '__main__':
    SeaCucumber().run()
```