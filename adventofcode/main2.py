from collections import *
import re

class Event:
    def __init__(self, data: str):
        pattern = r"[^[]*\[([^]]*)\]"
        m = re.search(pattern, data)
        self.date = m.group(1)
        pattern = r"#[0-9]*"
        m = re.search(pattern, data)
        self.id = int(m.group(0)[1:]) if m else None
        if self.id:
            self.is_asleep = None
        elif "asleep" in data:
            self.is_asleep = True
        else:
            self.is_asleep = False
    def __repr__(self):
        return f'date: {self.date}, status: {self.is_asleep}'
def main():
    with open('input.txt', 'r') as f:
        data = sorted(list(map(Event, f.read().splitlines())), key = lambda event: event.date)
        guards = defaultdict(lambda: [0]*60)
        guard = None
        # O(len(data))
        for event in data:
            if event.id:
                guard = event.id
            elif event.is_asleep:
                start = int(event.date.split()[1].split(':')[1])
                guards[guard][start] += 1
            else:
                end = int(event.date.split()[1].split(':')[1])
                guards[guard][end] -= 1
        for key in guards.keys():
            cnt = 0
            for i in range(len(guards[key])):
                cnt += guards[key][i]
                guards[key][i] = cnt
        sleepiest_guard = max(guards.keys(), key = lambda key: max(guards[key]))
        sleepiest_day = max(range(len(guards[sleepiest_guard])), key = lambda time: guards[sleepiest_guard][time])
        # O(len(guards) * 60) ~ O(len(data))
        return sleepiest_guard*sleepiest_day
if __name__ == "__main__":
    print(main())