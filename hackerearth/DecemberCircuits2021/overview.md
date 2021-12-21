December Circuits '21

Smallest path

Given an n-ary tree find the kth smallest element in simple path between nodes.

I don't know how to solve this problem

I was thinking to find the lowest common ancestor to get the simple path between two nodes and store the values in a set,
This would be QNlog(N) time complexity.  I would have to remove from set until I reach the 5th element.  There is no way to index into such a thing. 

This is one approach, but I think it requires a more complicated algorithm.  

I found there is a way to find lowest common ancestor in O(nlogn) time, but I don't know how much this even helps.  Cause
I still need to construct the simple path in O(n) time and add to a set, then I will get the result
but that is still the time complexity that is going to be too slow.  


Prerequisites:
Learn sparse tables
sparse table for range minimum query
Learn binary lifting for LCA
Learn this technique for LCA https://www.youtube.com/watch?v=dOAxrhAUIhA
==========================================================================================
A Planned Trip


I can't figure out probabilities at the moment.  


==========================================================================================

Sum of Cards

Solution: Greedy trick, just ceil(sum/X) 

```c++
int T, N, X;
cin>>T;
while (T--) {
    cin>>N>>X;
    int sum = 0;
    for (int i = 0;i<N;i++) {
        int a;
        cin>>a;
        sum += a;
    }
    sum = abs(sum);
    cout<<(sum+X-1)/X<<endl;
}
```

