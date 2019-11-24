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

for i in range(0,10):
    bandits.append(Bandit(mean=np.random.normal(mean_all_action,var_all_aciion), variance=var_single_game))

# create the game

greddy_game = Game(bandits)
output = 0
output_mean = []
output_variance = []

num_step = 1000
num_game = 2000

for i in range(0,num_bandits):
    output_mean.append(greddy_game.bandits[i].mean)
    output_variance.append(greddy_game.bandits[i].variance)
    print(" Arm {} mean = {}, variance = {}".format(i,output_mean[i],output_variance[i]))

output_std = np.sqrt(output_variance)

# show the 10 arm bandit
'''
fig, ax = plt.subplots()
ax.bar(np.arange(num_bandits), output_mean, yerr=output_std, align='center', alpha=0, ecolor='black', capsize=10)
plt.savefig('bar_plot_with_error_bars.png')
plt.show()
'''

for i in range(num_step):

print(output / 1000)
print(greddy_game.bandits[0].mean)