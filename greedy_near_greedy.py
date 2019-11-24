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
    greedy_history = []
    greedy_game = []
    greedy_correctness = []

    near_greedy_history = []
    near_greedy_game = []
    near_greedy_correctness = []

    near_greedy2_history = []
    near_greedy2_game = []
    near_greedy2_correctness = []

    for i in range(0,num_game):
        bandits = []
        for _ in range(0, num_bandits):
            bandits.append(Bandit(mean=np.random.normal(mean_all_action, var_all_aciion), variance=var_single_game))
        greedy_game.append(Game(bandits))

    for i in range(0, num_game):
        bandits = []
        for _ in range(0, num_bandits):
            bandits.append(Bandit(mean=np.random.normal(mean_all_action, var_all_aciion), variance=var_single_game))
        near_greedy_game.append(Game(bandits))

    for i in range(0, num_game):
        bandits = []
        for _ in range(0, num_bandits):
            bandits.append(Bandit(mean=np.random.normal(mean_all_action, var_all_aciion), variance=var_single_game))
        near_greedy2_game.append(Game(bandits))


    all_game_mean_1 = 0
    all_game_mean_2 = 0
    all_game_mean_3 = 0

    all_game_corr_1 = 0
    all_game_corr_2 = 0
    all_game_corr_3 = 0

    for i in range(num_step):
        for j in range(num_game):
            #print("game : {} , step : {}".format(j,i))
            greedy_game[j].sample_once(near_greedy=False)
            all_game_mean_1 = all_game_mean_1 + greedy_game[j].get_cur_mean()

            all_game_corr_1 = all_game_corr_1 + greedy_game[j].correct_count
            #print("cur_mean : {}".format(greedy_game[j].get_cur_mean()))
        #print(all_game_mean_1)
        print("greedy step {}".format(i))
        greedy_history.append(all_game_mean_1 / num_game)
        greedy_correctness.append(all_game_corr_1 / num_game)
        all_game_mean_1 = 0
        all_game_corr_1 = 0

    for i in range(num_step):
        for j in range(num_game):
            #print("game : {} , step : {}".format(j,i))
            near_greedy_game[j].sample_once(near_greedy=True,prop=0.01)
            all_game_mean_2 = all_game_mean_2 + near_greedy_game[j].get_cur_mean()

            all_game_corr_2 = all_game_corr_2 + near_greedy_game[j].correct_count
            #print("cur_mean : {}".format(greedy_game[j].get_cur_mean()))
        #print(all_game_mean_1)
        print("near greedy 1 step {}".format(i))
        near_greedy_history.append(all_game_mean_2 / num_game)
        near_greedy_correctness.append(all_game_corr_2 / num_game)
        all_game_mean_2 = 0
        all_game_corr_2 = 0

    for i in range(num_step):
        for j in range(num_game):
            #print("game : {} , step : {}".format(j,i))
            near_greedy2_game[j].sample_once(near_greedy=True,prop=0.1)
            all_game_mean_3 = all_game_mean_3 + near_greedy2_game[j].get_cur_mean()

            all_game_corr_3 = all_game_corr_3 + near_greedy2_game[j].correct_count
            #print("cur_mean : {}".format(greedy_game[j].get_cur_mean()))
        #print(all_game_mean_1)
        print("near greedy 2 step {}".format(i))
        near_greedy2_history.append(all_game_mean_3 / num_game)
        near_greedy2_correctness.append(all_game_corr_3 / num_game)
        all_game_mean_3 = 0
        all_game_corr_3 = 0

    plt.plot(greedy_correctness)
    plt.plot(near_greedy_correctness)
    plt.plot(near_greedy2_correctness)
    plt.savefig('allstep_percentage.png')
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
