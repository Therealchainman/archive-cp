
"""
The problem is asking to find the number of edges and the edges necessary to make the 
graph strongly connected component.  A strongly connected component is when a directed
graph can visit all nodes from any source node.  

1) Find the minimum number of edges
max(count of vertices with 0 indegee, count of vertices with 0 outdegree)
2) Then just iterate and connect the nodes 0 indegrees to those of 0 outdegrees, and 
if there is 1 more node with 0 indegree , can connect it to any other node.  

It is the same as New Flight Routes in cses 
"""