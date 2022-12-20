from collections import defaultdict, deque
import time

class ListNode:
    def __init__(self, val, index, next=None, prev = None):
        self.val = val
        self.index = index
        self.next = next
        self.prev = prev
    
    def __repr__(self):
        return f'Node: val: {self.val}, index: {self.index}, next: {self.next}'

def surroundings(node):
    print(node.prev.val, node.val, node.next.val)

def main():
    with open("input.txt", 'r') as f:
        data = list(map(int, f.read().splitlines()))
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
        node = head
        # surroundings(node)
        large = -9993
        smallest = -9993
        for i in range(len(data)):
            while node.index != i:
                node = node.next
            if node.val == 0: 
                # print(node.prev.val, node.next.val)
                continue
            if node.val == large:
                surroundings(node)
            smallest = min(smallest, node.val)
            # finding the next nodes
            cur = node
            if node.val > 0:
                for i in range(node.val):
                    cur = cur.next
                new_next = cur.next
               new_prev = cur
            else:
                for i in range(abs(node.val)):
                    cur = cur.prev
                new_next = cur
                new_prev = cur.prev

            if node.val == large:
                print(new_prev.val, new_next.val)
            
            # remove node from current position
            node.prev.next = node.next
            node.next.prev = node.prev
            if node.val == large:
                print(node.prev.val, node.prev.next.val, node.next.val)

            # insert node at new position
            node.prev = new_prev
            node.next = new_next
            if node.val == large:
                surroundings(node)
            
            new_prev.next = node
            new_next.prev = node
            if node.val == large:
                surroundings(node)
        res = 0
        while node.val != 0:
            node = node.next
        distance = 3000
        for i in range(distance+1):
            if i % 1000 == 0:
                print(node.val)
                res += node.val
            node = node.next
        return res

if __name__ == '__main__':
    start_time = time.perf_counter()
    print(main())
    end_time = time.perf_counter()
    print(f'Time Elapsed: {end_time - start_time} seconds')