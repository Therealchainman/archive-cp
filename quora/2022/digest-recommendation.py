"""
Copyright (C) 2022 Todd Chaney - All Rights Reserved
You may use, distribute and modify this code under the
terms of the XYZ license. Once I figure out licensing
"""

"""
assume I've read in the following data

n m => stories, users
n lines for stories each user created 
p q => user follows, story follows
p lines for user follows Ui Uj
q lines for story follows Ui Sk

The worse case time complexity for this algorithm is O(n*m^2)
"""
from collections import namedtuple
import unittest

class DigestRecommendation:
    def __init__(self):
        self.user_follows_user = []
        self.user_follows_story = []
        self.user_creates_story = []
        self.num_users = 0
        self.num_stories = 0

    def data_loader(self):
        n, m = map(int, input().split())
        self.num_users = m
        self.num_stories = n
        self.user_creates_story = [[0 for _ in range(n+1)] for _ in range(m+1)]
        self.user_follows_story = [[0 for _ in range(n+1)] for _ in range(m+1)]
        self.user_follows_user = [[0 for _ in range(m+1)] for _ in range(m+1)]
        for i in range(1,n+1):
            user = int(input())
            self.user_creates_story[user][i]=1
        p, q = map(int,input().split())
        for _ in range(p):
            ui, uj = map(int,input().split())
            self.user_follows_user[ui][uj]=1
        for _ in range(q):
            ui, sk = map(int, input().split())
            self.user_follows_story[ui][sk]=1

    def display(self, data):
        """
        The function to displays the result.
        """
        print(" ".join(map(lambda x: str(x.index),data)))

    def debug(self):
        """
        Code that was used for debugging
        """
        print("===useri follows users===")
        for i in range(1,self.num_users+1):
            print(f"useri={i} follows users={self.user_follows_user[i]}")
        print("===useri follows stories===")
        for i in range(1,self.num_users+1):
            print(f"useri={i} follows stories={self.user_follows_story[i]}")
        print("===useri creates stories===")
        for i in range(1,self.num_users+1):
            print(f"useri={i} creates stories={self.user_creates_story[i]}")

    def run(self):
        """
        The main function to run the algorithm.
        """
        self.data_loader()
        # self.debug()
        user_scores = [[0 for _ in range(self.num_users+1)] for _ in range(self.num_users+1)]
        for i in range(1,self.num_users+1):
            for j in range(1,self.num_users+1):
                if i==j: continue
                if self.user_follows_user[i][j]:
                    user_scores[i][j] = 3
                    continue
                for k in range(1,self.num_stories+1):
                    if self.user_follows_story[i][k] and self.user_creates_story[j][k]:
                        user_scores[i][j] = 2
                        break
                    if self.user_follows_story[i][k] and self.user_follows_story[j][k]:
                        user_scores[i][j] = 1
        story_scores = [[0 for _ in range(self.num_stories+1)] for _ in range(self.num_users+1)]
        for i in range(1,self.num_users+1):
            for k in range(1,self.num_stories+1):
                if self.user_creates_story[i][k]:
                    story_scores[i][k] = 2
                    continue
                if self.user_follows_story[i][k]:
                    story_scores[i][k] = 1
        StoryScore = namedtuple("StoryScore", ["score", "index"])
        # O(n*m^2 + m*nlogn)
        for i in range(1,self.num_users+1):
            digest_scores = []
            for k in range(1,self.num_stories+1):
                if self.user_follows_story[i][k] or self.user_creates_story[i][k]:
                    story_score = StoryScore(-1,k)
                    digest_scores.append(story_score)
                    continue
                score = 0
                for j in range(1,self.num_users+1):
                    score += user_scores[i][j]*story_scores[j][k]
                story_score = StoryScore(score, k)
                digest_scores.append(story_score)
            digest_scores.sort(key = lambda x: (-x.score, x.index))
            self.display(digest_scores[:3])


class TestDigestRecommendation(unittest.TestCase):
    def test_user_scores(self):
        """
        Test the user scores function
        """
        digestRecommendationSystem = DigestRecommendation()
        digestRecommendationSystem.data_loader()
        self.assertEqual(digestRecommendationSystem.user_score(1,1), 0)
        self.assertEqual(digestRecommendationSystem.user_score(1,2), 3) # if useri follows userj
        self.assertEqual(digestRecommendationSystem.user_score(1,5), 2) # if useri follows stories created by userj
        self.assertEqual(digestRecommendationSystem.user_score(1,3),0) # otherwise

    def test_story_scores(self):
        """
        Test the story scores function
        """
        digestRecommendationSystem = DigestRecommendation()
        digestRecommendationSystem.data_loader()
        self.assertEqual(digestRecommendationSystem.story_score(2,3), 2) # if useri created story k
        self.assertEqual(digestRecommendationSystem.story_score(3,4), 2) # if useri created story k
        self.assertEqual(digestRecommendationSystem.story_score(3,3), 1) # if user1 follows story k
        self.assertEqual(digestRecommendationSystem.story_score(3,7), 0) # if user1 follows story k
    

if __name__ == '__main__':
    digestRecommendationSystem = DigestRecommendation()
    digestRecommendationSystem.run()
    # unittest.main()
