import random
import tensorflow as tf

import games_data

print('Loading games...')
gamedata = games_data.GamesData('./users', './games', 'ledger.json', 'users_conv.json',
                                time_value_function=(lambda _: _), use_recent=True, min_games_played=5)
users = gamedata.get_users_conv()

# Steam ID for final output
your_steam_id = '76561197960287930'

dim_input = gamedata.games_count
dim_output = gamedata.games_count
dim_hidden1 = 2500
dim_hidden2 = 2500

batch_size = 512
epochs = 1000
lr = .1

x = tf.placeholder("float", [None, dim_input])
y = tf.placeholder("float", [None, dim_input])

weights1 = tf.Variable(tf.random_normal([dim_input, dim_hidden1]))
weights2 = tf.Variable(tf.random_normal([dim_hidden1, dim_hidden2]))
weights3 = tf.Variable(tf.random_normal([dim_hidden2, dim_input]))

biases1 = tf.Variable(tf.random_normal([dim_hidden1]))
biases2 = tf.Variable(tf.random_normal([dim_hidden2]))
biases3 = tf.Variable(tf.random_normal([dim_input]))

neural_net = tf.sigmoid(tf.add(tf.matmul(x, weights1), biases1))
neural_net = tf.sigmoid(tf.add(tf.matmul(neural_net, weights2), biases2))
neural_net = tf.sigmoid(tf.add(tf.matmul(neural_net, weights3), biases3))

# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_net, labels=y))
loss = tf.reduce_mean(tf.squared_difference(neural_net, y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train = optimizer.minimize(loss=loss)

# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(neural_net)), tf.argmax(y)), tf.float32))
accuracy = tf.reduce_mean(tf.squared_difference(neural_net, y))


class BatchFormatter:
    def __init__(self, users_conv, hidden_games):
        self.users_conv = users_conv
        self.hidden_games = hidden_games

        self.__read_count = 0
        # self.users_conv = list(filter(lambda games: self.__count_games_played(games) >
        # user_min_games_played, self.users_conv))

    # Sorry about this low quality method
    def __hide_random_games(self, games):
        games = games.copy()
        u_games = []
        for ind in range(0, len(games)):
            if games[ind] > 0:
                u_games.append(ind)

        for it in range(0, min(self.hidden_games, len(u_games))):
            games[u_games.pop(random.randrange(len(u_games)))] = 0
        return games

    def __get_next_user(self):
        self.__read_count += 1
        try:
            row = next(self.users_conv[1])
        except StopIteration:
            self.users_conv[0].seek(0)
            row = next(self.users_conv[1])
        # print(self.__read_count)
        return [float(r) for r in row]

    def __get_next_users(self, count):
        # return [eff_users[ind] for ind in range(eff_index, min(eff_index + count, len(eff_users)))]
        return [self.__get_next_user() for i in range(0, count)]

    def get_batch(self, count):
        usrs = list(self.__get_next_users(count))

        return {'x': [self.__hide_random_games(u) for u in usrs],
                'y': [u for u in usrs]}


batches = BatchFormatter(users_conv=users, hidden_games=1)

# print('Collected batches with ' + str(batches.size_train()) + ' training elements '
#                                                              'and ' + str(batches.size_test()) + ' testing elements.')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, epochs+1):
        batch = batches.get_batch(batch_size)
        # print('\tinput was x: ' + str(len(batch['x'])) + ' y: ' + str(len(batch['x'])))
        _, out = sess.run([train, neural_net], feed_dict={x: batch['x'], y: batch['y']})
        print('expected: ' + str(batch['x'][0]))
        print('actual: ' + str(out[0]))
        sloss = sess.run([loss], feed_dict={x: [batch['x'][0]], y: [out[0]]})
        print('single-loss: ' + str(sloss[0]))
        if epoch % 1 == 0:
            cost, acc = sess.run([loss, accuracy], feed_dict={x: batch['x'], y: batch['y']})
            print('Epoch ' + str(epoch) + ': Acc= ' + "{:.5f}".format(acc) + " Cost= " + "{:.5f}".format(cost))

    print('Train Epochs Finished')
