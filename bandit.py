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
        for bandit in bandits:
            self.bandits.append(bandit)

    def get_reward(self, arm=0):
        return self.bandits[arm].normal_sample()
