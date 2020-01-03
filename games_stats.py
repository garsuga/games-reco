# Used to provide insights into data for games

import numpy as np
import matplotlib.pyplot as plt

import games_data

gamedata = games_data.GamesData('users', 'games', 'ledger.json', 'users_conv.json', use_recent=True,
                                min_games_played=1)

print("Loading games")
games_by_user = np.genfromtxt('users_conv.json', dtype=float, delimiter=',')

print(games_by_user.shape)

print("Collecting sums")
sums = games_by_user.sum(axis=0)
print(sums.shape)

print("Sorting sums")
sums_sort = np.argsort(sums)
# sums = sums[sums_sort]

print("Collecting means")
means = games_by_user.mean(axis=0)
print(means.shape)

print("Sorting means by sums order")
means = means[sums_sort]

plt.title("Playtime by game")
plt.subplot(121)
plt.plot(sums, 'b-')

plt.subplot(122)
plt.plot(means, 'r.')

plt.show()

print("Normalizing data by percent total playtime per user (g = g / tot)")
sums_per_person = np.sum(games_by_user, axis=1)
sums_per_person = sums_per_person.reshape((sums_per_person.shape[0], 1))
norm_games = np.divide(games_by_user, sums_per_person)

norm_test = np.sum(norm_games, axis=1)
print(norm_test)

plt.title('Average percent playtime per person by game')
plt.plot(np.sort(np.mean(norm_games, axis=0)), 'b-')
plt.show()
