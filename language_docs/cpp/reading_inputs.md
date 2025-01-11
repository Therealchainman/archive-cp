# Reading inputs

Rule need to know for cin and getline(cin, s)

When you use getline after cin, you need to use cin.ignore() otherwise the first getline will return an empty string.
I think it is the end of the previous line. 

```cpp
int T, MOD;
string text;
cin >> T >> MOD;
cin.ignore();
getline(cin, text);
```