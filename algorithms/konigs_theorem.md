# Konig's Theorem

Kőnig's theorem: in a bipartite graph, the size of the maximum matching is equal to the size of the minimum vertex cover.

minimum vertex, means you take the smallest set of vertex that will cover every edge, that is all the edges will be incident to all the vertex in the set.

A tree is a bipartite graph

problem
https://leetcode.com/problems/maximum-students-taking-exam/description/

maximize students by finding the maximum independent set of vertices
The maximum independent set is the complement of the minimum vertex cover
In graph theory and independent set, anticlique is a set of vertices in a graph, no two of which are adjacent.  That is, it is a set of vertices such that for every two vertices in S, there is no edge connecting the two. Each edge has at most one endpoint in S. 
A set is independent if and only if it is a clique in the graph's complement. The size of an independent set is the number of vertices it contains. Independent sets have also been called "internally stable sets", of which "stable set" is a shortening.


https://codeforces.com/contest/1948/problem/GF