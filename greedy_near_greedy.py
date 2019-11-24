'''

Demostrate the greedy and near greedy method

'''

import numpy as np
import matplotlib.pyplot as plt
from bandit import *

# create 10 bandit for the game following
# mean of the true action value : 0
# variance of true action value : 1

mean_all_action = 0
var_all_aciion = 1
var_single_game = 1

num_bandits = 10

bandits = []

for i in range(0,num_bandits):
    bandits.append(Bandit(mean=np.random.normal(mean_all_action,var_all_aciion), variance=var_single_game))

# create the game

greedy_game = Game(bandits)
output = [[] for _ in range(num_bandits)]
output_mean = []
output_variance = []

num_step = 1000
num_game = 2000


for i in range(0,num_bandits):
    output_mean.append(greedy_game.bandits[i].mean)
    output_variance.append(greedy_game.bandits[i].variance)
    print(" Arm {} mean = {}, variance = {}".format(i,output_mean[i],output_variance[i]))

output_std = np.sqrt(output_variance)

# show the 10 arm bandit
'''
fig, ax = plt.subplots()
ax.bar(np.arange(num_bandits), output_mean, yerr=output_std, align='center', alpha=0, ecolor='black', capsize=10)
plt.savefig('bar_plot_with_error_bars.png')
plt.show()
'''

def multiple_game_greedy_action():
    history = []
    games = []

    for i in range(0,num_game):
        bandits = []
        for _ in range(0, num_bandits):
            bandits.append(Bandit(mean=np.random.normal(mean_all_action, var_all_aciion), variance=var_single_game))
        cur_game = Game(bandits)
        games.append(Game(bandits))


    all_game_mean = 0

    for i in range(num_step):
        for j in range(num_game):
            #print("game : {} , step : {}".format(j,i))
            games[j].sample_once(near_greedy=False)
            all_game_mean = all_game_mean + games[j].get_cur_mean()
            #print("cur_mean : {}".format(games[j].get_cur_mean()))
        #print(all_game_mean)
        print("step {}".format(i))
        history.append(all_game_mean / num_game)
        all_game_mean = 0

    plt.plot(history)
    plt.show()

def one_game_greedy_action():
    history = []
    for i in range(num_step):
        greedy_game.sample_once(near_greedy=False)
        history.append(greedy_game.get_cur_mean())

    print(greedy_game.mean)
    print(greedy_game.count)
    print(greedy_game.get_total_reward() / num_step)

    plt.plot(history)
    plt.show()


#one_game_greedy_action()
multiple_game_greedy_action()
