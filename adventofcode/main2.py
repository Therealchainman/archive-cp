import time

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

def main():
    with open("input.txt", 'r') as f:
        decryption_key = 811589153
        data = list(map(lambda num: int(num)*decryption_key, f.read().splitlines()))
        cdl = CircularDoublyLinkedList()
        node = cdl.build(data) # constructs doubly linked list our of the array and returns the head
        rounds = 10
        for _ in range(rounds):
            for i in range(cdl.size):
                while node.index != i:
                    node = node.next
                if node.val%(cdl.size-1) == 0: continue
                cdl.remove(node)
                new_prev, new_next = cdl.locate(node)
                cdl.insert(node, new_next, new_prev)
        res = 0
        distance = [1000, 2000, 3000]
        for i in distance:
            while node.val != 0:
                node = node.next
            node = cdl.forward(node, i)
            res += node.val
        return res

if __name__ == '__main__':
    start_time = time.perf_counter()
    print(main())
    end_time = time.perf_counter()
    print(f'Time Elapsed: {end_time - start_time} seconds')