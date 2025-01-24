#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <string>
#include <type_traits>
using namespace std;
#define endl '\n'

// Helper trait to detect container types
template <typename T, typename _ = void>
struct is_container : std::false_type {};

template <typename T>
struct is_container<T, std::void_t<decltype(std::declval<T>().begin()),
                                   decltype(std::declval<T>().end())>> : std::true_type {};

// Base case: Debug single value (non-container types)
template <typename T>
typename std::enable_if<!std::is_same<T, std::string>::value && !std::is_class<T>::value>::type
debug(const T &x) {
    std::cerr << x;
}

template <typename T1, typename T2>
void debug(const pair<T1, T2> &p) {
    cerr << "(";
    debug(p.first);
    cerr << ", ";
    debug(p.second);
    cerr << ")";
}

// Debug for containers (vector, set, map, etc.)
template <typename T>
typename std::enable_if<is_container<T>::value>::type
debug(const T &container) {
    cerr << "{";
    bool first = true;
    for (const auto &elem : container) {
        if (!first) cerr << ", ";
        first = false;
        debug(elem);
    }
    cerr << "}";
}

// Variadic Debug to handle multiple arguments
template <typename T, typename... Args>
void debug(const T &first, const Args &...rest) {
    debug(first);
    cerr << " ";
    debug(rest...);
}

// Debug for `std::string` (special case to avoid treating it as a container)
void debug(const std::string &s) {
    std::cerr << "\"" << s << "\"";
}