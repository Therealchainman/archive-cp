# Competitions

This will be updated during the year as the season plays out with code for problems I've solved.  

These are solved because they give real time practice with working with algorithms, data structures, 
mathematics, and problem solving in a fun and relaxing environment.

Can expect to find solutions written in the following programming languages, C++, python (legacy), and Rust (soon).

### Setup python virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Reference for running C++ code

```sh
g++ main.cpp -o main
```

### Monitoring resources on ubuntu

1. htop: This is a interactive process viewer for the terminal. It allows you to view detailed information about the processes running on your system, including their CPU and memory usage. To install htop, open a terminal and type sudo apt-get install htop. Then, to run htop, simply type htop in the terminal.
1. top: This is a similar tool to htop, but it is not as interactive. To run top, simply type top in the terminal.
1. free: This command displays information about the amount of free and used memory in the system. To run it, type free in the terminal.
1. df: This command displays information about the amount of free space on your system's disks. To run it, type df in the terminal.
1. lscpu: This command displays information about the CPU in your system, including the number of cores, the architecture, and the clock speed. To run it, type lscpu in the terminal.
1. lshw: This command displays detailed information about all of the hardware in your system. To run it, type sudo lshw in the terminal. You may need to install the lshw package first by running sudo apt-get install lshw.

## Usage of jit compiler with pypy

Supposed to prevent the memory error, prevents recursion function requiring lots of memory, but can slow down recursive function slightly. 

```py
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')
```

Setting max_unroll_recursion to -1 essentially disables recursion unrolling, meaning that the JIT compiler will not attempt to unroll recursive function calls at all. This can be useful in cases where unrolling causes performance degradation due to increased memory usage or when recursion depth is unknown or unpredictable. However, disabling unrolling may also result in slower execution for recursive code.

## MacOS Set up

On MacOS to compile using the Apple Clang compiler, at version 16.0.0, with target arm64-app-darwin24.1.0 you need to specify the standard for C++, cause it uses a legacy C++ standard by default. 

```sh
g++ -std=c++20 <program file> -o main
```

## Template files

The intended purpose of the following templates is for platforms like Codeforces, and AtCoder. 
There exists template file that is used to compile C++ code, `template.cpp`
And there is a template for python code, `template.py`

## Debugging on MacOS

```cpp
g++ -g <program file> -o main
lldb main
run
```