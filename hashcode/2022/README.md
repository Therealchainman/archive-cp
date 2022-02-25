# Google Hashcode 2022 Summary

## Strategy

1. Write the input/ouput
1. Write a python script that will score current run
1. Write simple greedy solution
1. Analyze data to get a better idea of possible solutions
1. graph the upper bound to find areas of greatest improvement

Scheduling + Assignment Optimization Problem

## Simple Greedy Algorithm with Heap Datastructure

This solution is very slow and it will take too long on any dataset other than a and b.

While I have projects assigned in a min heap of assigned projects and have not assigned all 
projects. I loop through the projects each time and assign contributors based on those that 
are closest to the required skill and if I cannot assign the contributors for a project it
does not assign that project.  This could happen because some contributors with skills necessary
for role is not available and assigned to another project that is in the heap datastructure. 
Or maybe no contributor has the appropriate level, in which case this project requires leveling up.
Either way these projects are skipped.  Then the heap datastructure is increased in size. 
And we pop out elements from the heap datastructure for the next earliest day.  These contributors
may level up and also they will be free and available again.  

This is also not great because it has to iterate through all skills and add the contributor back to all them.

Each time we loop through projects, loop through roles, find contributor and so on.  

## Can I Parallelize

In addition there is no good way to parallelize this algorithm.  We can't possible break it down into
chunks and compute just one, because if a contributor is assigned to a project in one chunk.  Then we 
must communicate to the other chunk that this is so, so it doesn't double assign a contributor.  

And it doesn't make sense to split it into contributor chunks.  The problem with this is that we may be missing
a required contributor to make a project possible.  

So breaking down into project chunk or contributor chunk doesn't work.  Which removes the possible of improving
the speed of this algorithm with parallelization/concurrency.  

This means there is no solution for the slow computing time on these datasets

## Can I remove the min heap datastructure

One idea to improve the algorithm is to remove the min heap datastructure, but this creates other problems

## Can I remove the method of skills mapping to all contributors with a sortedList



## Rewrite in C++ 

This is a simple solution to just give a speedup to the algorithm.  This might be worthwhile
to implement and potentially the most speedup with fewest hours of coding and figuring out

## Improve the order of projects

I can improve the preferred order of projects by taking some heuristic such as 
score / duration, this way I prefer projects that have a good score to duration ratio. 
Most likely I can finish these,  also useful to have a sort on the earliest time to start

last_start_day_bonus = max(0, best_day - duration)

I can play around with these a little, but not really until I have an algorithm that runs 
in a reasonable runtime. 

## Is it possible to use bipartite matching

So why would I consider this problem as having a possible bipartite matching.  
The reason is that this optimization problem is a bit similar in literature to the 
assignment problem, which can be solved efficiently with bipartite matching.  

I need to conduct more research but initially it seems rather hard to implement because
the time element involved and how contributors can be available at different times

_______________________________________________________________________________________

Ideas of how to improve the greedy solution, and make it faster

I seemed to have made some minor improvements by updating the ordered_projects list every iteration 

Find the earliest day on which we can start the project

prioritize contributors that are available most recent and skill_level >= required_skill_level 
then prioritize based on skill_level is closest to the required_skill_level


loop through the projects
- loop through the roles required for that project
-- loop through the contributors 


loop through the projects
- loop through the contributors
-- loop through the roles required for that project

unsolved: How can I update the skill level for a contributor so that it is visible to a project

another idea is to perform these in batches that way I can update the skill level for contributors in each batch. 

[    batch1     ]   

[    batch2      ]

[    batch3     ]


last_start_day_bonus = max(0, best_day - duration)

I might be able to use this data to make better decision with the batches


any ideas with queues? 

What if projects are placed in a queue based on the last_start_day_bonus and maybe the score

potentially want the things with the best score

losing out on the possiblity of running projects in parallel

How would bipartite matching work to solve this problem? 
