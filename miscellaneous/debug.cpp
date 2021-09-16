#include <bits/stdc++.h>
using namespace std;

void trimRightTrailingSpaces(string &input) {
    input.erase(find_if(input.rbegin(), input.rend(), [](int ch) {
        return !isspace(ch);
    }).base(), input.end());
}

void trimLeftTrailingSpaces(string &input) {
    input.erase(input.begin(), find_if(input.begin(), input.end(), [](int ch) {
        return !isspace(ch);
    }));
}

template <class T>
T stringToValue(const string &str);

template<>
auto stringToValue(const string &str) -> int {
    return stoi(str);
}

void getSingleVectorPreprocess(string &input, string &forStringstream) {
    trimLeftTrailingSpaces(input);
    trimRightTrailingSpaces(input);
    auto len = input.length();
    if (len < 2) {
        return;
    }
    forStringstream = input.substr(1, len - 2);
}

template <class T>
vector<T> getSingleVector(string &input, char delim, bool doPreprocess) {
    stringstream ss;
    vector<T> ans;
    if (doPreprocess) {
        string forStringstream;
        getSingleVectorPreprocess(input, forStringstream);
        if (forStringstream.empty()) {
            return ans;
        }
        ss.str(forStringstream);
    } else {
        ss.str(input);
    }
    string item;
    while (true) {
        if (!getline(ss, item, delim)) {
            break;
        }
        if (delim == ' ' && item == "") {
            continue;
        }
        ans.push_back(stringToValue<T>(item));
    }
    return ans;
}

template <class T>
vector<vector<T>> getDoubleVector(string &input, char delim) {
    vector<vector<T>> ans;
    trimLeftTrailingSpaces(input);
    trimRightTrailingSpaces(input);
    char left_bracket = input[0];
    auto len = input.length();
    if (len < 2) {
        return ans;
    }
    char right_bracket = input[len - 1];
    string input1 = input.substr(1, len - 2);
    if (input1.empty()) {
        return ans;
    }
    int first_left_pos = input1.find(left_bracket);
    int first_right_pos = input1.find(right_bracket);
    while (true) {
        if (first_left_pos > first_right_pos || first_left_pos == string::npos || first_right_pos == string::npos) {
            break;
        }
        string s = input1.substr(first_left_pos + 1, first_right_pos - first_left_pos - 1);
        ans.push_back(getSingleVector<int>(s, delim, false));
        first_left_pos = input1.find(left_bracket, first_right_pos + 1);
        first_right_pos = input1.find(right_bracket, first_right_pos + 1);
    }
    return ans;
}