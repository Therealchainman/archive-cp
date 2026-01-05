// #include <iostream>
// #include <vector>
// #include <set>
// #include <map>
// #include <utility>
// #include <string>
// #include <type_traits>
// using namespace std;
// #define endl '\n'

// template <typename T, typename _ = void>
// struct is_container : std::false_type {};

// template <typename T>
// struct is_container<T, std::void_t<decltype(std::declval<T>().begin()),
//                                    decltype(std::declval<T>().end())>> : std::true_type {};

// // Forward declaration to handle nested containers
// template <typename T>
// typename std::enable_if<is_container<T>::value && !std::is_same<T, std::string>::value>::type
// debug(const T &container);

// // Special case: std::string (treat as scalar, not container)
// void debug(const std::string &s) {
//     std::cerr << "\"" << s << "\"";
// }

// // Base Case: Primitives, Structs, Pairs (Non-containers)
// // We removed !std::is_class<T> so custom structs are allowed
// template <typename T>
// typename std::enable_if<!is_container<T>::value && !std::is_same<T, std::string>::value>::type
// debug(const T &x) {
//     std::cerr << x;
// }

// // Special Case: Pairs
// template <typename T1, typename T2>
// void debug(const pair<T1, T2> &p) {
//     cerr << "(";
//     debug(p.first);
//     cerr << ", ";
//     debug(p.second);
//     cerr << ")";
// }

// // tuple support
// template <std::size_t I = 0, typename... Ts>
// std::enable_if_t<I == sizeof...(Ts)>
// debug_tuple_elements(const std::tuple<Ts...>&) {}

// template <std::size_t I = 0, typename... Ts>
// std::enable_if_t<I < sizeof...(Ts)>
// debug_tuple_elements(const std::tuple<Ts...>& t) {
//     if constexpr (I > 0) {
//         std::cerr << ", ";
//     }
//     debug(std::get<I>(t));
//     debug_tuple_elements<I + 1>(t);
// }

// template <typename... Ts>
// void debug(const std::tuple<Ts...>& t) {
//     std::cerr << "(";
//     debug_tuple_elements(t);
//     std::cerr << ")";
// }

// // Special Case: Vector<bool> (std quirk)
// void debug(const std::vector<bool> &vb) {
//     std::cerr << "{";
//     for (size_t i = 0; i < vb.size(); ++i) {
//         if (i > 0) std::cerr << ", ";
//         std::cerr << (vb[i] ? "true" : "false");
//     }
//     std::cerr << "}";
// }

// // Container Logic
// template <typename T>
// typename std::enable_if<is_container<T>::value && !std::is_same<T, std::string>::value>::type
// debug(const T &container) {
//     cerr << "{";
//     bool first = true;
//     for (const auto &elem : container) {
//         if (!first) cerr << ", ";
//         first = false;
//         debug(elem);
//     }
//     cerr << "}";
// }

// // Variadic Debug
// template <typename T, typename... Args>
// void debug(const T &first, const Args &...rest) {
//     debug(first);
//     cerr << " ";
//     debug(rest...);
// }

// // TreeNode definition
// struct TreeNode {
//     int val;
//     TreeNode *left;
//     TreeNode *right;
//     TreeNode() : val(0), left(nullptr), right(nullptr) {}
//     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
//     TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
// };

// // Helper function to recursively print the tree in a hierarchical format.
// void printTree(TreeNode* node, const string &prefix, bool isLeft) {
//     std::cerr << prefix;
//     std::cerr << (isLeft ? "├── " : "└── ");
//     if (node == nullptr) {
//         std::cerr << "null" << endl;
//         return;
//     }
//     std::cerr << node->val << endl;
//     // Only print children if at least one exists
//     if (node->left != nullptr || node->right != nullptr) {
//         string newPrefix = prefix + (isLeft ? "│   " : "    ");
//         printTree(node->left, newPrefix, true);
//         printTree(node->right, newPrefix, false);
//     }
// }

// // Overload for TreeNode* to integrate with the debug library.
// void debug(TreeNode* root) {
//     if (root == nullptr) {
//         std::cerr << "null" << endl;
//         return;
//     }
//     // Print root without any branch symbol.
//     std::cerr << root->val << endl;
//     // If at least one child exists, print them with appropriate branches.
//     if (root->left != nullptr || root->right != nullptr) {
//         printTree(root->left, "", true);
//         printTree(root->right, "", false);
//     }
// }

#include <deque>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::pair<T1, T2>& pair);

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec);

template <typename T, size_t SZ>
std::ostream& operator<<(std::ostream& out, const std::array<T, SZ>& arr);

template <typename T, typename C, typename A>
std::ostream& operator<<(std::ostream& out, const std::set<T, C, A>& set);

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::multiset<T>& set);

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::map<T1, T2>& map);

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::pair<T1, T2>& pair) {
    return out << '(' << pair.first << ", " << pair.second << ')';
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
    if (vec.empty()) {
        out << "[]";
        return out;
    }
    out << '[';
    for (int i = 0; i < vec.size() - 1; i++) {
        out << vec[i] << ", ";
    }
    return out << vec.back() << ']';
}

template <typename T, size_t SZ>
std::ostream& operator<<(std::ostream& out, const std::array<T, SZ>& arr) {
    if (SZ == 0) {
        out << "[]";
        return out;
    }
    out << '[';
    for (int i = 0; i < SZ - 1; i++) {
        out << arr[i] << ", ";
    }
    return out << arr.back() << ']';
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::deque<T>& deq) {
    if (deq.empty()) {
        out << "[]";
        return out;
    }
    out << '[';
    for (int i = 0; i < deq.size() - 1; i++) {
        out << deq[i] << ", ";
    }
    return out << deq.back() << ']';
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::unordered_map<T1, T2>& map) {
    out << '{';
    for (auto it = map.begin(); it != map.end(); it++) {
        std::pair<T1, T2> element = *it;
        out << element.first << ": " << element.second;
        if (std::next(it) != map.end()) {
            out << ", ";
        }
    }
    return out << '}';
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::map<T1, T2>& map) {
    out << '{';
    for (auto it = map.begin(); it != map.end(); it++) {
        std::pair<T1, T2> element = *it;
        out << element.first << ": " << element.second;
        if (std::next(it) != map.end()) {
            out << ", ";
        }
    }
    return out << '}';
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::unordered_set<T>& set) {
    out << '{';
    for (auto it = set.begin(); it != set.end(); it++) {
        T element = *it;
        out << element;
        if (std::next(it) != set.end()) {
            out << ", ";
        }
    }
    return out << '}';
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::multiset<T>& set) {
    out << '{';
    for (auto it = set.begin(); it != set.end(); it++) {
        T element = *it;
        out << element;
        if (std::next(it) != set.end()) {
            out << ", ";
        }
    }
    return out << '}';
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::unordered_multiset<T>& set) {
    out << '{';
    for (auto it = set.begin(); it != set.end(); it++) {
        T element = *it;
        out << element;
        if (std::next(it) != set.end()) {
            out << ", ";
        }
    }
    return out << '}';
}

template <typename T, typename C, typename A>
std::ostream& operator<<(std::ostream& out, const std::set<T, C, A>& set) {
    out << '{';
    for (auto it = set.begin(); it != set.end(); it++) {
        T element = *it;
        out << element;
        if (std::next(it) != set.end()) {
            out << ", ";
        }
    }
    return out << '}';
}

// Source: https://stackoverflow.com/a/31116392/12128483
template <typename Type, unsigned N, unsigned Last>
struct TuplePrinter {
    static void print(std::ostream& out, const Type& value) {
        out << std::get<N>(value) << ", ";
        TuplePrinter<Type, N + 1, Last>::print(out, value);
    }
};

template <typename Type, unsigned N>
struct TuplePrinter<Type, N, N> {
    static void print(std::ostream& out, const Type& value) {
        out << std::get<N>(value);
    }
};

template <typename... Types>
std::ostream& operator<<(std::ostream& out, const std::tuple<Types...>& value) {
    out << '(';
    TuplePrinter<std::tuple<Types...>, 0, sizeof...(Types) - 1>::print(out, value);
    return out << ')';
}