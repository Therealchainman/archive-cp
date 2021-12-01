
# Part 1

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

# Part 2

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