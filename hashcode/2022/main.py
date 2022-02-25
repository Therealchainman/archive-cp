from sortedcontainers import SortedList
from collections import defaultdict
from heapq import heappop, heappush, heapify

class Contributor:
    def __init__(self, name, num_skills):
        self.name = name
        self.num_skills = num_skills
        self.skills = {}
        
    def __repr__(self):
        return f"contributor name: {self.name}, number of skills: {self.num_skills}, skills: {self.skills}\n"

class Skill:
    def __init__(self, name, skill_level):
        self.name = name
        self.skill_level = skill_level
        
    def __repr__(self):
        return f"(skill name: {self.name}, skill level: {self.skill_level})"
    
class Project:
    def __init__(self, name, duration, score, best_day, num_roles):
        self.name = name
        self.duration = duration
        self.score = score
        self.best_day = best_day
        self.num_roles = num_roles
        self.max_skill_level = 0
        self.skills = []
        
    def __repr__(self):
        return f"project name: {self.name}, number of skills: {self.num_roles}, duration: {self.duration}, score: {self.score}, best_day: {self.best_day}, skills: {self.skills}\n"

class AssignedProject:
    def __init__(self, project_name, contributors, learnings):
        self.project_name = project_name
        self.contributors = contributors
        self.learnings = learnings
        
    def __lt__(self, other):
        return False
        
    def __repr__(self):
        return f"project name: {self.project_name}, contributors: {self.contributors}, learnings: {self.learnings}"
    
class Solver:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    def data_loader(self):
        with open(f'inputs/{self.dataset_name}.in') as f:
            self.num_contributors, self.num_projects = map(int, f.readline().split())
            # print(self.num_contributors, self.num_projects)
            self.contributors = {}
            for _ in range(self.num_contributors):
                contributor_name, num_skills = f.readline().split()
                num_skills = int(num_skills)
                self.contributors[contributor_name] = Contributor(contributor_name, num_skills)
                for _ in range(num_skills):
                    skill_name, skill_level = f.readline().split()
                    skill_level = int(skill_level)
                    self.contributors[contributor_name].skills[skill_name] = Skill(skill_name, skill_level)
            self.projects = {}
            for _ in range(self.num_projects):
                project_name, duration, score, best_day, num_roles = f.readline().split()
                duration = int(duration)
                score = int(score)
                best_day = int(best_day)
                num_roles = int(num_roles)
                self.projects[project_name] = Project(project_name, duration, score, best_day, num_roles)
                for _ in range(num_roles):
                    skill_name, skill_level = f.readline().split()
                    skill_level = int(skill_level)
                    self.projects[project_name].max_skill_level = max(self.projects[project_name].max_skill_level, skill_level)
                    self.projects[project_name].skills.append(Skill(skill_name, skill_level))
            
    def run(self):
        self.data_loader()
        ordered_projects = []
        self.skills_contributors = {}
        self.contributor_skill_levels = defaultdict(dict)
        unique_skills = set()
        for contrib in self.contributors.values():
            for skill in contrib.skills.values():
                unique_skills.add(skill.name)
        
        # PREPROCESSING THE CONTRIBUTORS SKILL LEVELS IN ALL SKILLS AND SKILLS WITH CONTRIBUTORS
        for skill in unique_skills:
            self.skills_contributors[skill] = SortedList(key=lambda x: x[0])
            for contrib in self.contributors.values():
                if skill in contrib.skills:
                    self.skills_contributors[skill].add((contrib.skills[skill].skill_level, contrib.name))
                    self.contributor_skill_levels[contrib.name][skill] = contrib.skills[skill].skill_level
                else:
                    self.skills_contributors[skill].add((0, contrib.name))
                    self.contributor_skill_levels[contrib.name][skill] = 0
        
        for proj in self.projects.values():
            # skip skills that can't be completed unless you level up many times
            if proj.max_skill_level > 10: continue
            ordered_projects.append(proj)
        
        ordered_projects.sort(key=lambda x: (x.best_day, -x.score))
        
        # TODO: queue up the people that are available with the current 
        # For each role in a project, find the best person available with the appropriate skill and skill level
        # how to query the available contributors
        
        day = 0
        min_heap = []
        completed_projects = set()
        total_score = 0
        assigned_projects = []
        while len(completed_projects) < self.num_projects:
            # print(f"heap datastructure with assigned projects: {min_heap}")
            if min_heap:
                day = min_heap[0][0]
            while min_heap and min_heap[0][0] == day:
                _, completed_project = heappop(min_heap)

                # UPDATE THE LEARNINGS
                for contributor_name, skill_name in completed_project.learnings:
                    self.contributor_skill_levels[contributor_name][skill_name] += 1


                # ADD CONTRIBUTORS AS AVAILABLE
                for contributor_name in completed_project.contributors:
                    for skill_name in unique_skills:
                        self.skills_contributors[skill_name].add((self.contributor_skill_levels[contributor_name][skill_name],contributor_name))

            for proj in ordered_projects:
                if proj.name in completed_projects: continue
                assigned_contributors = []
                seen_contributors = set()
                learnings = []
                can_assign = True
                # print(f"project skills: {proj.skills}")
                for role in proj.skills:
                    skill_name, level = role.name, role.skill_level
                    index = self.skills_contributors[skill_name].bisect_left((level, 0))
                    if index == len(self.skills_contributors[skill_name]): 
                        can_assign = False
                        break
                    
                    # contributor_name = self.skills_contributors[skill_name][index][0]
                    # contributor_skill_level = self.contributor_skill_levels[contributor_name][skill_name]
                    # self.skills_contributors[skill_name].remove((contributor_skill_level, contributor_name))
                    # if level == contributor_skill_level:
                    #     learnings.append((contributor_name, skill_name))
                    
                    # TODO: ADD A SET FOR CHECKING IF THE CONTRIBUTOR IS ASSIGNED
                    len_before = len(assigned_contributors)
                    for i in range(index, len(self.skills_contributors[skill_name])):
                        contributor_name = self.skills_contributors[skill_name][i][1]
                        if contributor_name in seen_contributors: continue
                        assigned_contributors.append(contributor_name)
                        seen_contributors.add(contributor_name)
                        # print(contributor_name, level, self.contributor_skill_levels[contributor_name][skill_name])
                        if level == self.contributor_skill_levels[contributor_name][skill_name]:
                            learnings.append((contributor_name, skill_name))
                        break
                    len_after = len(assigned_contributors)
                    if len_before == len_after:
                        can_assign = False
                        break

                if not can_assign: continue

                # ASSIGN PROJECT
                assigned_project = AssignedProject(proj.name, assigned_contributors, learnings)
                # print(f"project that can be assigned: {assigned_project}")
                
                # MIN HEAP DATASTRUCTURE FOR ASSIGNED PROJECT
                heappush(min_heap, (day + proj.duration, assigned_project))
                
                # SIMULATING THE PROJECT SCORE 
                project_score = max(0, proj.score - (max(0,(day + proj.duration)-proj.best_day)))
                total_score += project_score
                
                # print(len(assigned_contributors), len(unique_skills))
                
                # REMOVE THE CONTRIBUTORS NO LONGER AVAILABLE
                for contrib in assigned_contributors:
                    for skill in unique_skills:
                        contrib_skill_level = self.contributor_skill_levels[contrib][skill]
                        self.skills_contributors[skill].remove((contrib_skill_level, contrib))
                
                # print(f"After removing contributors working on the project: {self.skills_contributors}")
                
                # UPDATE THE COMPLETED PROJECTS
                completed_projects.add(proj.name)
                assigned_projects.append(assigned_project)
            # print(f"heap datastructure with assigned projects: {min_heap}")
            # print(f"progress: {len(completed_projects)}/{len(ordered_projects)}")
            
            # EXITS WHEN IT NO LONGER HAS PROJECTS THAT CAN BE ASSIGNED
            if not min_heap:
                break
        # print(f"score estimation: {total_score}")

        # WRITE TO OUTPUT FILE
        with open(f"outputs/{self.dataset_name}.out", "w") as f:
            f.write(str(len(assigned_projects)) + '\n')
            for proj in assigned_projects:
                f.write(proj.project_name + '\n')
                f.write(" ".join(proj.contributors) + '\n')
        
        # RETURNS THE TOTAL SCORE
        return total_score


# Get the upper_bound for the score for each dataset
# assume that I complete all projects by the best day
def upper_bound():
    scores = {}
    total_score = 0
    total_actual_score = 0
    for dataset in ['a','b','c','d','e','f']:
        sol = Solver()
        sol.data_loader(f"{dataset}.in")
        actual_score = sol.run(dataset)
        upper_bound_score = 0
        for project in sol.projects.values():
            upper_bound_score += project.score
        print(f"dataset: {dataset}, upper_bound_score: {upper_bound_score}, actual_score: {actual_score}, offset: {upper_bound_score - actual_score}")
        total_score += upper_bound_score
        total_actual_score += actual_score
    print(f"total_socre upper_bound score: {total_score}, total actual score: {total_actual_score}, total offset: {total_score - total_actual_score}")

if __name__ == '__main__':
  upper_bound()