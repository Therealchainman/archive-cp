# TREE ROTATIONS

## left rotation

returning the right node because that will be the new right node for the grand parent node Look at DSW algorithm for example of this

```py
def left_rotation(node):
    prev_node = node
    node = node.right
    prev_node.right = node.left
    node.left = prev_node
    return node
```

## right rotation

returning the new parent is helpful for when creating a backbone tree.  

```py
def right_rotation(node):
    prev_node = node
    node = node.left
    node_left_right = node.right
    node.right = prev_node
    prev_node.left = node_left_right
    return node
```

## DSW(Day, Stout, and Warren) algorithm to balance binary search tree

This creates a vine or backbone which is right leaning with right rotations and moving to the new parent node, until there is no left child. 
The second phase it performs left rotations to balance the tree

```py
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        # PHASE 1: CREATE THE RIGHT LEANING VINE/BACKBONE 
        def right_rotation(node):
            prev_node = node
            node = node.left
            node_left_right = node.right
            node.right = prev_node
            prev_node.left = node_left_right
            return node
        def create_vine(grand):
            tmp = grand.right
            cnt = 0
            while tmp:
                if tmp.left:
                    tmp = right_rotation(tmp)
                    grand.right = tmp
                else:
                    cnt += 1
                    grand = grand.right
                    tmp = tmp.right
            return cnt
        grand_parent = TreeNode()
        grand_parent.right = root
        # count number of nodes
        n = create_vine(grand_parent)
        # PHASE 2: LEFT ROTATIONS TO GET BALANCED BINARY SEARCH TREE
        # height_perfect_balanced_tree
        h = int(math.log2(n + 1))
        # needed_nodes_perfect_balanced_tree
        m = pow(2, h) - 1
        excess = n - m 
        def left_rotation(node):
            prev_node = node
            node = node.right
            prev_node.right = node.left
            node.left = prev_node
            return node
        def compress(grand_parent, cnt):
            node = grand_parent.right
            while cnt > 0:
                cnt -= 1
                node = left_rotation(node)
                grand_parent.right = node
                grand_parent = node
                node = node.right
        compress(grand_parent, excess)
        while m > 0:
            m >>= 1
            compress(grand_parent, m)
        return grand_parent.right
```