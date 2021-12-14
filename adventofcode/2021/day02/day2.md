Python and C++ 

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