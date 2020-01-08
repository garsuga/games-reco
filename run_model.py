import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import urllib.request
import math

import games_model_2 as gm2
import games_data


def plot_games(indices, values, games):
    plt.figure(figsize=(18, 36))
    plot_num = 0
    for ind in indices:
        game = games[plot_num]

        try:
            fs = urllib.request.urlopen(game.img_logo_url)
        except:
            fs = urllib.request.urlopen('http://media.steampowered.com/steamcommunity'
                                        '/public/images/apps/10/6b0312cda02f5f777efa2f3318c307ff9acafbb5.jpg')
        img = Image.open(fs)
        rows = int(math.ceil(math.sqrt(len(indices))))
        cols = int(math.ceil(len(indices) / rows))

        val = values[plot_num]

        plot_num += 1
        plt.subplot(rows * 100 + cols * 10 + plot_num)
        plt.imshow(img)
        plt.title('#' + str(plot_num) + ': ' + game.name + ' (' + "{:.5f}".format(val) + ')')
    plt.show()


file_store = 'users_conv_norm_filt.csv'
file_conv_ledger = 'users_conv_norm_filt_ledger.csv'

gamedata = games_data.GamesData('users', 'games', 'ledger.json', 'users_conv.csv', True, 1)

data = None

if not os.path.exists(file_store):
    # Load games from file
    games_by_user = np.genfromtxt('users_conv.csv', dtype=float, delimiter=',')

    # Create sums for playtime per person
    sums_per_person = np.sum(games_by_user, axis=1)
    # Make into 2 dimensional array of same dimensions
    sums_per_person = sums_per_person.reshape((sums_per_person.shape[0], 1))
    # Divide each element of each row by the total for that row
    norm_games = np.divide(games_by_user, sums_per_person)

    # Get sums for total playtime by game
    sums = games_by_user.sum(axis=0)

    # Get upper 500 partition in asc order based on gameplay sums
    top_games_ind = np.argpartition(sums, -500)[-500:]
    # Get original values and form a set of sorted indices and then reorder the partitioned indexes using those.
    # Reverse to provide desc order
    top_games_ind = top_games_ind[np.argsort(sums[top_games_ind])][::-1]

    # Build ledger to go back to gamedata ledger
    np.savetxt(file_conv_ledger, top_games_ind)

    # Switch axes of the normalized games by users array and chose only rows within the top 500 games.
    # Flipped axes to original position
    norm_games_top = np.transpose(np.transpose(norm_games)[top_games_ind])
    # Filter out users whose playtime is now 0 with only top 500 games
    data = norm_games_top[np.argwhere(np.sum(norm_games_top, axis=1) > 0).flatten()]

    # Save for future use
    np.savetxt(file_store, data)
else:
    # Load previous (and much smaller) values
    data = np.genfromtxt(file_store)

# The data ledger is a mapping from an index within the top 500 to an original index for the gamedata ledger
data_ledger = np.genfromtxt(file_conv_ledger, dtype=int)

# Create model with 2 hidden layers and equal input/output dims
model = gm2.GamesModel(0.001, [data.shape[1], 25, data.shape[1]])

# labels = data.copy()
# labels = np.array([hide_random_games(labels[i], 1) for i in range(0, labels.shape[0])])

model.open_session()
model.train(data=data, epochs=1000, batch_size=256, print_interval=100)

# Get recommendations from my personal steam id
old_vals = gamedata.get_user_games_from_file('76561198067935522')
# old_vals = gamedata.get_user_games_from_file('76561197960287930')
old_vals = old_vals / np.sum(old_vals)
old_vals = old_vals[data_ledger]

reco = model.recommend([old_vals])[0]

# get 8 max element indices
top_ind = np.argpartition(reco, -8)[-8:]
# sort them by value in asc order and reverse to end with desc order
top_ind = top_ind[np.argsort(reco[top_ind])][::-1]

print(model.test_loss([reco], [old_vals]))








































































plot_games(top_ind, reco[top_ind], [gamedata.get_game_from_index(data_ledger[i]) for i in top_ind])


