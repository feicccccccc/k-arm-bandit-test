'''

An exercise base on RL book by Sutton

'''

import numpy as np


class Bandit:
    def __init__(self, mean=0, variance=1):
        self.mean = mean
        self.variance = variance

    def normal_sample(self):
        return np.random.normal(self.mean, self.variance)


class Game:

    def __init__(self, bandits=[]):
        self.bandits = []
        self.results = [[] for _ in range(len(bandits))]
        self.mean = [0 for _ in range(len(bandits))]
        self.count = [0 for _ in range(len(bandits))]
        self.correct_count = 0
        for bandit in bandits:
            self.bandits.append(bandit)
        self.optimal_action = 0
        max_reward = -10
        for i, bandit in enumerate(self.bandits):
            #print("{} {}".format(i, bandit))
            if bandit.mean > max_reward:
                max_reward = bandit.mean
                self.optimal_action = i

    def get_reward(self):
        return (self.sample_action(), self.bandits[self.sample_action()].normal_sample())

    def sample_action(self):
        action_candidate = np.argwhere(self.mean == np.amax(self.mean))
        action_candidate = np.random.choice(action_candidate.flatten())
        return action_candidate

    def sample_once(self,near_greedy = True, prop = 0.5):
        if near_greedy:
            sample = np.random.uniform(0, 1)
            if sample < prop:
                action = np.random.randint(10)
            else:
                action = self.sample_action()
        else:
            action = self.sample_action()
        if action == self.optimal_action:
            self.correct_count =  self.correct_count + 1
        reward = self.bandits[action].normal_sample()

        self.results[action].append(reward)
        self.mean[action] = np.mean(self.results[action])
        self.count[action] = self.count[action] + 1

    def get_total_reward(self):
        total_reward = 0
        for i in range(len(self.bandits)):
            total_reward = total_reward + self.count[i] * self.mean[i]
        return total_reward

    def get_cur_mean(self):
        total_reward = self.get_total_reward()
        return total_reward / sum(self.count)
