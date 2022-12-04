from collections import *
class Event:
    def __init__(self, data: str):
        index = data.find(']')
        self.date = data[1:index]
        self.status = data[index+2:]
    def __repr__(self):
        return f'date: {self.date}, status: {self.status}'
def main():
    with open('input.txt', 'r') as f:
        data = sorted(list(map(Event, f.read().splitlines())), key = lambda event: event.date)
        guards = defaultdict(lambda: [0]*60)
        guard = None
        # O(len(data))
        for event in data:
            if 'Guard' in event.status:
                guard = int(event.status.split()[1][1:])
            elif event.status == 'falls asleep':
                start = int(event.date.split()[1].split(':')[1])
                guards[guard][start] += 1
            else:
                end = int(event.date.split()[1].split(':')[1])
                guards[guard][end] -= 1
        for key, vals in guards.items():
            cnt = 0
            for i in range(len(vals)):
                cnt += vals[i]
                vals[i] = cnt
            guards[key] = vals
        sleepiest_guard = max(guards.keys(), key = lambda key: sum(guards[key]))
        sleepiest_day = max(range(len(guards[sleepiest_guard])), key = lambda time: guards[sleepiest_guard][time])
        # O(len(guards) * 60) ~ O(len(data))
        return sleepiest_guard*sleepiest_day
if __name__ == "__main__":
    print(main())