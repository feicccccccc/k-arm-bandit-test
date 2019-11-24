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
    bandits = []

    def __init__(self, bandits=[]):
        self.results = [[] for _ in range(len(bandits))]
        self.mean = [0 for _ in range(len(bandits))]
        for bandit in bandits:
            self.bandits.append(bandit)

    def get_reward(self):
        return (self.sample_action(), self.bandits[self.sample_action()].normal_sample())

    def sample_action(self):
        action_candidate = np.argwhere(self.mean == np.amax(self.mean))
        action_candidate = np.random.choice(action_candidate.flatten())
        return action_candidate

    def sample_once(self):
        action = self.sample_action()
        reward = self.bandits[action].normal_sample()

        self.results[action].append(reward)
        self.mean[action] = np.mean(self.results[action])

        test2 = 0