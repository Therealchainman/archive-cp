# TREE TRAVERSALS

## inorder traversal

recursion generator function

```py
def inorder(node):
    if node:
        yield from inorder(node.left)
        yield node.val
        yield from inorder(node.right)
```

dfs stack iterative approach with generator

```py
def inorder(node):
    stack = []
    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        yield node
        node = node.right
```