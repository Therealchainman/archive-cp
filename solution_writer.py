"""
Creates a solution template
"""

import sys

problem_counts = {"atcoder": 7, "codeforces": 6, "leetcode": 4}

def create(contest, name, number):
    if contest == "atcoder":
        path = f"{contest}/{name}{number}.md"
        sys.stdout = open(path, 'w')
        print(f"# Atcoder Beginner Contest {number}")
        print()
        for _ in range(problem_counts[contest]):
            print("## ")
            print()
            print("### Solution 1: ")
            print()
            print("```py")
            print()
            print("```")
            print()
    elif contest == "codeforces":
        pass
    elif contest == "leetcode":
        path = f"{contest}/{name}{number}.md"
        sys.stdout = open(path, 'w')
        print(f"# Leetcode Weekly Contest {number}")
        print()
        for _ in range(problem_counts[contest]):
            print("## ")
            print()
            print("### Solution 1: ")
            print()
            print("```py")
            print()
            print("```")
            print()
    sys.stdout.close()

if __name__ == '__main__':
    contest, name, number = "atcoder", "beginner", 314
    create(contest, name, number)