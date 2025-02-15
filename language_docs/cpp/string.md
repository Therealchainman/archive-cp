# String

## Split a string with delimiter

This is an example function that takes a string where each word is separated by a space.  And using the istringstream iss(s) and getline with specifying the delimiter as single space you can create a vector of the words.

```cpp
vector<string> process(const string& s, char delimiter = ' ') {
    vector<string> ans;
    istringstream iss(s);
    string word;
    while (getline(iss, word, delimiter)) ans.emplace_back(word);
    return ans;
}
```