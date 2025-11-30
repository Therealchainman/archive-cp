# European Championship 2024

## F. Dating

### Solution 1: 

```py

```

## G. Scooter

### Solution 1:  algorithm

```py
def drive(i):
    return f"DRIVE {i + 1}"

PICK = "PICKUP"
DROP = "DROPOFF"

def main():
    n = int(input())
    classes = input()
    prof = input()
    # index where you have M but need C
    math = [i for i in range(n) if classes[i] == 'C' and prof[i] == 'M']
    # index where you have C but need M
    comp = [i for i in range(n) if classes[i] == 'M' and prof[i] == 'C']
    # index where you need C
    needC = [i for i in range(n) if classes[i] == 'C' and prof[i] == "-"]
    # index where you need M
    needM = [i for i in range(n) if classes[i] == 'M' and prof[i] == "-"]
    # pickups
    pickups = [i for i in range(n) if classes[i] == "-" and prof[i] != "-"]
    ans = []
    for p in pickups:
        ans.append(drive(p))
        ans.append(PICK)
        cur = prof[p]
        while cur is not None:
            if cur == "C":
                if math:
                    ans.append(drive(math.pop()))
                    ans.append(DROP)
                    ans.append(PICK)
                    cur = "M"
                elif needC:
                    ans.append(drive(needC.pop()))
                    ans.append(DROP)
                    cur = None
                else:
                    if ans[-1] == PICK: ans.pop()
                    cur = None
            elif cur == "M":
                if comp:
                    ans.append(drive(comp.pop()))
                    ans.append(DROP)
                    ans.append(PICK)
                    cur = "C"
                elif needM:
                    ans.append(drive(needM.pop()))
                    ans.append(DROP)
                    cur = None
                else:
                    if ans[-1] == PICK: ans.pop()
                    cur = None
    print(len(ans))
    print("\n".join(ans))

if __name__ == '__main__':
    main()
```

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```