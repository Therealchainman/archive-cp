# Circular Doubly Linked List Data Structure

I know this implementation is slower than implementing a deque in python, but it is still quite fast for the advent of code problems. I also wanted to practice implementing a circular doubly linked list.

It has the ability to perform
- construct circular doubly linked list from an array of integers
- iterate forward to locate the new_next and new_prev for the movement of the node
- remove a node
- insert a node
- move forward through the nodes to return a node that is at some displacement
- These functions work using modular arithmetic to speed it up and avoid iterating multiple times through the circular linked list

```py
"""
constructs circular doubly linked list from an array of integers
"""
class CircularDoublyLinkedList:
    def build(self, data):
        self.size = len(data)
        head = None
        tail = None
        for i, num in enumerate(data):
            node = ListNode(num, i)
            if head is None:
                head = node
                tail = node
            else:
                node.prev = tail
                tail.next = node
                tail = node
        head.prev = tail
        tail.next = head
        return head

    def locate(self, node):
        cur = node
        cnt = node.val%(self.size-1)
        for _ in range(cnt):
            cur = cur.next
        return (cur, cur.next)

    def display(self, node, rng):
        for _ in range(rng):
            node = node.next

    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def insert(self, node, new_next, new_prev):
        node.prev = new_prev
        node.next = new_next
        new_prev.next = node
        new_next.prev = node

    def forward(self, node, i):
        for _ in range(i%self.size):
            node = node.next
        return node

"""
List Node class for a circular doubly linked list with methods for removing, inserting, and locating nodes
"""
class ListNode:
    def __init__(self, val, index, next=None, prev = None):
        self.val = val
        self.index = index
        self.next = next
        self.prev = prev

    def __repr__(self):
        return f'val: {self.val}'
```