# Ordered Set

## Implemented with Red Black Tree

Similar to a `std::set`, but allows for efficient order statistics operations.

size()
insert(const T& key)
erase(const T& key)
find_by_order(size_t k) // 0-based k-th smallest, nullptr if out of range

```cpp
template <typename T, typename Compare = std::less<T>>
class OrderedSet {
    enum Color { RED, BLACK };
    struct Node {
        T key;
        Color color;
        size_t size;
        Node *left, *right, *parent;
        Node(const T& k)
            : key(k), color(RED), size(1), left(nullptr), right(nullptr), parent(nullptr) {}
    };

    Node* root_;
    Compare cmp_;

    // Utility: get size of node (0 if null)
    size_t node_size(Node* x) const {
        return x ? x->size : 0;
    }
    void update_size(Node* x) {
        if (x)
            x->size = 1 + node_size(x->left) + node_size(x->right);
    }

    // Left rotate around x
    void left_rotate(Node* x) {
        Node* y = x->right;
        x->right = y->left;
        if (y->left) y->left->parent = x;
        y->parent = x->parent;
        if (!x->parent)
            root_ = y;
        else if (x == x->parent->left)
            x->parent->left = y;
        else
            x->parent->right = y;
        y->left = x;
        x->parent = y;
        // update sizes
        update_size(x);
        update_size(y);
    }

    // Right rotate around x
    void right_rotate(Node* x) {
        Node* y = x->left;
        x->left = y->right;
        if (y->right) y->right->parent = x;
        y->parent = x->parent;
        if (!x->parent)
            root_ = y;
        else if (x == x->parent->right)
            x->parent->right = y;
        else
            x->parent->left = y;
        y->right = x;
        x->parent = y;
        // update sizes
        update_size(x);
        update_size(y);
    }

    // BST insert + RB fixup
    void insert_fixup(Node* z) {
        while (z->parent && z->parent->color == RED) {
            Node* gp = z->parent->parent;
            if (z->parent == gp->left) {
                Node* y = gp->right;
                if (y && y->color == RED) {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    gp->color = RED;
                    z = gp;
                } else {
                    if (z == z->parent->right) {
                        z = z->parent;
                        left_rotate(z);
                    }
                    z->parent->color = BLACK;
                    gp->color = RED;
                    right_rotate(gp);
                }
            } else {
                Node* y = gp->left;
                if (y && y->color == RED) {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    gp->color = RED;
                    z = gp;
                } else {
                    if (z == z->parent->left) {
                        z = z->parent;
                        right_rotate(z);
                    }
                    z->parent->color = BLACK;
                    gp->color = RED;
                    left_rotate(gp);
                }
            }
        }
        root_->color = BLACK;
    }

    // Transplant u -> v
    void transplant(Node* u, Node* v) {
        if (!u->parent)
            root_ = v;
        else if (u == u->parent->left)
            u->parent->left = v;
        else
            u->parent->right = v;
        if (v)
            v->parent = u->parent;
    }

    Node* minimum(Node* x) {
        while (x->left) x = x->left;
        return x;
    }

    Node* find_node(const T& key) const {
        Node* x = root_;
        while (x) {
            if (cmp_(key, x->key))
                x = x->left;
            else if (cmp_(x->key, key))
                x = x->right;
            else
                return x;
        }
        return nullptr;
    }

    // Delete fixup
    void delete_fixup(Node* x, Node* x_parent) {
        while ((x != root_) && (!x || x->color == BLACK)) {
            if (x_parent && x == x_parent->left) {
                Node* w = x_parent->right;
                if (w && w->color == RED) {
                    w->color = BLACK;
                    x_parent->color = RED;
                    left_rotate(x_parent);
                    w = x_parent->right;
                }
                if ((!(w->left) || w->left->color == BLACK) &&
                    (!(w->right) || w->right->color == BLACK)) {
                    if(w) w->color = RED;
                    x = x_parent;
                    x_parent = x_parent->parent;
                } else {
                    if (!(w->right) || w->right->color == BLACK) {
                        if(w->left) w->left->color = BLACK;
                        w->color = RED;
                        right_rotate(w);
                        w = x_parent->right;
                    }
                    if(w) w->color = x_parent->color;
                    x_parent->color = BLACK;
                    if(w->right) w->right->color = BLACK;
                    left_rotate(x_parent);
                    x = root_;
                    break;
                }
            } else if(x_parent) {
                Node* w = x_parent->left;
                if (w && w->color == RED) {
                    w->color = BLACK;
                    x_parent->color = RED;
                    right_rotate(x_parent);
                    w = x_parent->left;
                }
                if ((!(w->left) || w->left->color == BLACK) &&
                    (!(w->right) || w->right->color == BLACK)) {
                    if(w) w->color = RED;
                    x = x_parent;
                    x_parent = x_parent->parent;
                } else {
                    if (!(w->left) || w->left->color == BLACK) {
                        if(w->right) w->right->color = BLACK;
                        w->color = RED;
                        left_rotate(w);
                        w = x_parent->left;
                    }
                    if(w) w->color = x_parent->color;
                    x_parent->color = BLACK;
                    if(w->left) w->left->color = BLACK;
                    right_rotate(x_parent);
                    x = root_;
                    break;
                }
            } else break;
        }
        if (x) x->color = BLACK;
    }

public:
    OrderedSet() : root_(nullptr) {}
    ~OrderedSet() {
        // TODO: recursively delete nodes to avoid memory leak
    }

    size_t size() const {
        return node_size(root_);
    }

    void insert(const T& key) {
        Node* z = new Node(key);
        Node* y = nullptr;
        Node* x = root_;
        while (x) {
            y = x;
            if (cmp_(z->key, x->key))
                x = x->left;
            else if (cmp_(x->key, z->key))
                x = x->right;
            else { // key already exists
                delete z;
                return;
            }
        }
        z->parent = y;
        if (!y)
            root_ = z;
        else if (cmp_(z->key, y->key))
            y->left = z;
        else
            y->right = z;
        // Update sizes up the chain
        Node* p = z;
        while (p) {
            update_size(p);
            p = p->parent;
        }
        insert_fixup(z);
    }

    void erase(const T& key) {
        Node* z = find_node(key);
        if (!z) return;
        Color y_orig = z->color;
        Node* x = nullptr;
        Node* x_parent = nullptr;
        if (!z->left) {
            x = z->right;
            x_parent = z->parent;
            transplant(z, z->right);
        } else if (!z->right) {
            x = z->left;
            x_parent = z->parent;
            transplant(z, z->left);
        } else {
            Node* y = minimum(z->right);
            y_orig = y->color;
            x = y->right;
            if (y->parent == z) {
                x_parent = y;
            } else {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
                x_parent = y->parent;
            }
            transplant(z, y);
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
            update_size(y);
        }
        // Update sizes up from x_parent
        for (Node* p = x_parent; p; p = p->parent)
            update_size(p);
        if (y_orig == BLACK)
            delete_fixup(x, x_parent);
        delete z;
    }

    // Number of elements strictly less than key
    size_t order_of_key(const T& key) const {
        size_t cnt = 0;
        Node* x = root_;
        while (x) {
            if (cmp_(key, x->key)) {
                x = x->left;
            } else {
                cnt += node_size(x->left) + (cmp_(x->key, key) ? 1 : 0);
                x = (cmp_(x->key, key) ? x->right : nullptr);
            }
        }
        return cnt;
    }

    // 0-based k-th smallest, nullptr if out of range
    const T* find_by_order(size_t k) const {
        Node* x = root_;
        while (x) {
            size_t ls = node_size(x->left);
            if (k < ls)
                x = x->left;
            else if (k == ls)
                return &x->key;
            else {
                k -= ls + 1;
                x = x->right;
            }
        }
        return nullptr;
    }

    class iterator {
        Node* node_;
    public:
        using value_type = T;
        using reference = T&;
        using pointer = T*;
        using iterator_category = forward_iterator_tag;
        using difference_type = ptrdiff_t;

        explicit iterator(Node* n): node_(n) {}
        iterator& operator++() {
            if (!node_) return *this;
            if (node_->right) {
                node_ = node_->right;
                while (node_->left) node_ = node_->left;
            } else {
                Node* p = node_->parent;
                while (p && node_ == p->right) {
                    node_ = p;
                    p = p->parent;
                }
                node_ = p;
            }
            return *this;
        }
        reference operator*() const { return node_->key; }
        pointer operator->() const { return &node_->key; }
        bool operator==(const iterator& o) const { return node_ == o.node_; }
        bool operator!=(const iterator& o) const { return node_ != o.node_; }
    };

    iterator begin() const {
        Node* n = root_;
        while (n && n->left) n = n->left;
        return iterator(n);
    }
    iterator end() const { return iterator(nullptr); }
};
```