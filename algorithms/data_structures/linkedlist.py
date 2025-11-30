"""
Implementation of a linked list datastructure that can be used when I need to check contains, add, delete from linked list
"""

class Node:
    def __init__(self, val=0, next_node=None):
        self.val = val
        self.next = next_node
    
class LinkedList:
    def __init__(self):
        self.head = Node()
    
    def add(self, val: int) -> None:
        if self.contains(val): return
        node = self.head
        while node.next:
            node = node.next
        node.next = Node(val)
    
    def remove(self, val: int) -> None:
        node = self.head
        while node.next:
            if node.next.val == val:
                node.next = node.next.next
                break
            node=node.next
        
    def contains(self, val: int) -> None:
        node = self.head.next
        while node:
            if node.val == val: return True
            node=node.next
        return False

"""
Doubly Linked List implemented from scratch using head node.
Also this has a cursor node to keep track of current node that a pointer is located at.  
"""

class Node:
    def __init__(self, val='', prev_node=None, next_node=None):
        self.val = val
        self.next = next_node
        self.prev = prev_node
    
class DoublyLinkedList:
    def __init__(self):
        self.head = Node() # head node
        self.cursor_node = self.head
        
    def add(self, text: str) -> None:
        node = self.cursor_node
        node_after = node.next
        for ch in text:
            node.next = Node(ch,node)
            node = node.next
        node.next = node_after
        if node_after:
            node_after.prev = node
        self.cursor_node = node
        
    def remove(self, num: int) -> int:
        node = self.cursor_node
        node_after = node.next
        counter = 0
        while counter < num:
            if node.val == '': break
            node = node.prev
            counter += 1
        node.next = node_after
        if node_after:
            node_after.prev = node
        self.cursor_node = node
        return counter

    def moveLeft(self, num: int) -> str:
        node = self.cursor_node
        for _ in range(num):
            if node.val == '': break
            node = node.prev
        self.cursor_node = node
        left_elements = []
        for _ in range(10):
            if node.val == '': break
            left_elements.append(node.val)
            node=node.prev
        return ''.join(reversed(left_elements))
    
    def moveRight(self, num: int) -> str:
        node = self.cursor_node
        for _ in range(num):
            if not node.next: break
            node = node.next
        self.cursor_node = node
        left_elements = []
        for _ in range(10):
            if node.val == '': break
            left_elements.append(node.val)
            node=node.prev
        return ''.join(reversed(left_elements))