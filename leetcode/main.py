class Solution:
    def survivedRobotsHealths(self, positions: List[int], healths: List[int], directions: str) -> List[int]:
        n = len(positions)
        robots = sorted(range(n), key = lambda i: positions[i])
        stack = []
        for i in robots:
            if directions[i] == 'L':
                while stack and directions[stack[-1]] == 'R' and healths[i] > 0:
                    idx = stack.pop()
                    if healths[idx] < healths[i]:
                        healths[i] -= 1
                        healths[idx] = 0
                    elif healths[idx] == healths[i]:
                        healths[i] = healths[idx] = 0
                    elif healths[idx] > healths[i]:
                        healths[i] = 0
                        healths[idx] -= 1
                    if healths[idx] > 0:
                        stack.append(idx)
            if healths[i] > 0:
                stack.append(i)
        return filter(lambda x: x > 0, healths)