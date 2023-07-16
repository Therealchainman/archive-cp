# Google Coding Competitions

This will be updated during the year as the season plays out with code for problems I've solved.  

## code jam

Code Jam is Google's longest running global coding competition, where programmers of all levels put their skills to the test. Competitors work their way through a series of online algorithmic puzzles to earn a spot at the World Finals, all for a chance to win the championship title and $15,000.

## kick start

Kick Start is a global online coding competition, consisting of three-hour rounds of a variety of algorithmic challenges designed by Google engineers. Participants can compete in one or all online rounds held throughout the year, and will have the opportunity to develop and grow their programming abilities while getting a glimpse into the technical skills needed for a career at Google (top participants may be invited to interview at Google).

## hash code 

Hash Code is a team programming competition, organized by Google, for students and professionals around the world. You pick your team and programming language and we pick an engineering problem for you to solve. This year’s contest kicks off with an Online Qualifications, where your team can compete virtually from wherever you’d like, alongside your virtual Hub. Top teams will then be invited to compete from our virtual World Finals.

### Setup conda environment

The conda environment has updated python version of 3.10

```sh
conda env create -f environment.yaml
conda activate programming
```

Some reason it installed the wrong python so I had to manually conda install with

```sh
conda install python=3.10
```

A simpler installation and install packages as needed is to just run the following using python 3.11

```sh
conda create --name programming python=3.11
```

```sh
g++ main.cpp -o main
```

Way to read in int fast in c++

```c++
inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}
```

### How to export a conda environment

```sh
conda env export > environment.yaml
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

https://kenkoooo.com/atcoder/#/table/