from collections import Counter, defaultdict
def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        folder_sizes = Counter()
        curDir = ['/']
        adj_list = defaultdict(list)
        for line in data: # O(len(data))
            if line.startswith('$'): # user command in terminal
                if line == '$ cd ..': # go up one directory
                    if len(curDir) > 0:
                        curDir.pop()
                elif line == '$ cd /': # root directory
                    curDir = ['/']
                elif 'cd' in line: # go to child directory
                    directory = line.split()[-1]
                    folder = '.'.join(curDir + [directory])
                    curDir.append(folder)
            else:
                file_size, dir_name = line.split()
                if file_size == 'dir':
                    print(curDir)
                    adj_list['.'.join(curDir)].append('.'.join(curDir + [dir_name]))
                else:
                    file_size = int(file_size)
                    # O(number of characters in curDir and file_name)
                    folder_sizes['.'.join(curDir)] += file_size # assign file size to file
        print(folder_sizes)
        print(adj_list)
        def postorder(node: str) -> int:
            print(node)
            for child in adj_list[node]:
                folder_sizes[node] += postorder(child)
            return folder_sizes[node]
        root = '/'
        postorder(root)
        threshold = 100000
        return sum([folder_size for folder_size in folder_sizes.values() if folder_size <= threshold])
if __name__ == "__main__":
    print(main())