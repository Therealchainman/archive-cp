# 8. String to Integer (atoi)

### Solution: string parsing function

```c++
int myAtoi(string s) {
    long long res = 0;
    int start = find_if(s.begin(),s.end(),[](const auto& a) {
        return a=='+' || a=='-' || isdigit(a) || isalpha(a) || a=='.';
    })-s.begin();
    int sign = s[start]=='-' ? -1 : 1;
    start += (s[start]=='-' || s[start]=='+');
    for (int i = start;i<s.size();i++) {
        if (res>INT_MAX) break;
        if (isdigit(s[i])) {
            res = (res*10 + (s[i]-'0'));
        } else {break;}
    }
    res *= sign;
    if (res>0) {
        return min(res, (long long)INT_MAX);
    }
    return max(res, (long long)INT_MIN);
}
```

## Solution: Finite State Machine 

```py
class Solution:
    def myAtoi(self, s: str) -> int:
        
        def trim_integer(val):
            if val > 2**31 -1:
                return 2**31-1
            elif val < -2**31:
                return -2**31
            return val
        
        # FINITE STATE MACHINE
        states = {
            1: {'blank': 1, 'sign': 2, 'digit': 3},
            2: {'digit': 3},
            3: {'digit': 3}
        }
        
        num = 0
        # START IN THE INIT STATE OF FSM
        cstate = sign = 1
        for ch in s:
            # FIND THE TRANSITION 
            transition = 'else'
            if ch==' ':
                transition = 'blank'
            elif ch in '-+':
                transition = 'sign'
            elif ch in '0123456789':
                transition = 'digit'
            # FIND IF THE TRANSITION IS POSSIBLE IN CURRENT STATE
            if transition not in states[cstate]: break
                
            # UPDATE THE CURRENT STATE OF THE FSM
            cstate = states[cstate][transition]
                
            # PROCESS THE INPUT ACCORDING TO THE STATE OF THE FSM
            if cstate == 2:
                sign = 1 if ch=='+' else -1
            elif cstate == 3:
                num = num*10 + int(ch)
                
        return trim_integer(sign*num)
```

## Solution 3:  string manipulation + lstrip + string slice

```py
class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.lstrip()
        sign = 1
        if not s: return 0
        if s[0] in '-+':
            sign = -1 if s[0]=='-' else 1
            s = s[1:]
        s = s.lstrip('0')
        result = 0
        for ch in s[:11]:
            if ch not in string.digits:
                break
            result = (result*10) + ord(ch)-ord('0')
        return min(2**31-1, max(-2**31,sign*result))
```