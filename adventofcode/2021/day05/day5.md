
# Part 1

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

# Part 2

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