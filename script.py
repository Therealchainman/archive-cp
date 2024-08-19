import os

# Directory where the files are located
directory = os.getcwd()
print(directory)

# Filter for files with a specific extension, e.g., '.txt'
files = [file for file in os.listdir(directory) if file >= "beginner350.md"]
print(files)

output_file = "abc350_400.md"

with open(output_file, 'w') as outfile:
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as infile:
            contents = infile.read()
            outfile.write(contents)
            outfile.write('\n')  # Add a newline between files
