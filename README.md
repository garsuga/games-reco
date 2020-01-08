# games-reco
Recommendations for Steam games generated based on user activity.
Attemps to generate game recommendations using Steam activity on a simple neural network made in Tensorflow.

## Usage
Requires data from https://github.com/garsuga/steam-api-crawler. 
The `users/` and `games/` directories from running `steam_spider.py` should be present in the working directory.

The model can be run by executing `run_model.py` or by referencing the GamesModel class in `game_model_2.py`.

## Creation Process And Notes
The hypothesis that I worked under was that if a user played similar games to a test user, then the test user would be more likely to
play other games that were present in the original user but not the test user. That is, if person A plays mostly the same games as person B,
then person A is more likely to play games that are present in person B's history than a random game.

In order to format this into a trainable model, I continued on that idea and assumed that if person A can be like person B and that makes them
more likely to play games from person B, then the games that person B plays are somehow related to each other and should be correlated to
be present with each other in a user's history.

The labels would be the user's history, represented as a set of numbers where each index is a specific game. The input data would be almost
the same set, but with one non-zero value removed. The goal of the model's training would be to fill in the missing value reliably. During
testing, the input data would be the complete array and the output would hopefully be the original array with new, connnected games added.
These new games would be the recommendations.

When originally starting work on games-reco, I had a dataset of 75,000 users and 22,000 games. 
The input values were either integer amounts of minutes played or decimal values representing each game's playtime as a percent of the player's total playtime.
After removing users without any time in any game (private profiles, private game times, and users who actually did not play anything)
the remaining dataset was only 8,500 users and 22,000 games.

The original model had many issues training and would have extremely high loss in both training and testing. I tried many solutions and
eventually determined that because of the very large arrays and the very low amount of non-zero values present, any small change to the weights
and biases by the optimizer would result in a very large change in loss. Basically, the data was too sparse for the model. Additionally,
there were not as many users as I would like to train a model with as many inputs.

To fix these issues I first restricted the games axis of my data table to the top 500 games by playtime. That was done by just dropping any
columns for games whose sums were not within the top 500. The users were then filtered again; this time users without playtime in the top
500 games were removed. Users whose data were now empty were extremely rare (less than 10), leaving almost all of the 8,500 users and displaying that
an overwhelmingly high amount of playtime in the top 500 vs. the bottom 21,500.

The remaining data, 8,500 users and 500 games, minimized loss effectively but the model had serious overfitting options, even with a single
hidden layer of size 25. This leads me to the final remarks of where this model is at. The main issue I see is that the loss function punishes
the neural network finding the correct combinations of games in almost every case. 

Imagine you input an array [0, .1, 0, .9]. The model has positively correlated index 0, 1, and 3 with each other through training. Seeing
each of these present, it adds to index 0 in the resulting array, as expected, but also adds to index 1 because 3 was present and vis versa.
This would actually result in raising the loss of that training step, even though it would be correct if index 1 or 3 was hidden in the input.
Ideally, loss would be calculated ignoring any indices whose input was not 0. I have yet to find an efficient way to do this in Tensorflow.

On top of that, I just neglected to put an activation function on any of the hidden layers which may have helped with the overfitting.

In conclusion, this model is far from ideal and incomplete.
