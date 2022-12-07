from collections import Counter
def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        file_sizes = Counter()
        folders = set([''])
        curPath = []
        for line in data:
            if line.startswith('$'):
                if line == '$ cd ..':
                    if len(curPath) > 0:
                        curPath.pop()
                elif line == '$ cd /':
                    curPath = []
                elif 'cd' in line:
                    directory = line.split()[-1]
                    curPath.append(directory)
                    folders.add('/'.join(curPath))
            else:
                file_size, file_name = line.split()
                if file_size == 'dir': continue
                file_size = int(file_size)
                file_sizes['/'.join(curPath + [file_name])] = file_size
        disk_space = 70000000
        required_unused_space = 30000000
        root = ''
        folder_sizes = Counter()
        for folder in folders:
            folder_size = sum([fsize for file, fsize in file_sizes.items() if file.startswith(folder)])
            folder_sizes[folder] = folder_size
        used_space_by_fs = folder_sizes[root] 
        unused_space = disk_space - used_space_by_fs
        needed_space_to_free = required_unused_space - unused_space
        return min([size for size in folder_sizes.values() if size >= needed_space_to_free])
if __name__ == "__main__":
    print(main())