import tensorflow as tf


class GamesModel:
    def __init__(self, learning_rate, dims):
        self.num_layers = len(dims)
        self.dims = dims

        self.learning_rate = learning_rate

        self.model = self.__model()

        self.session = None

    # Returns (model, feed_dict(empty))
    def __model(self):
        x = tf.placeholder('float', [None, self.dims[0]], name='x')
        y = tf.placeholder('float', [None, self.dims[-1]], name='y')

        weights = [
            tf.Variable(tf.sigmoid(tf.random_normal([self.dims[i], self.dims[i + 1]])))
            for i in range(0, len(self.dims) - 1)
        ]

        biases = [
            tf.Variable(tf.sigmoid(tf.random_normal([self.dims[i + 1]])))
            for i in range(0, len(self.dims) - 1)
        ]

        nn = None
        for i in range(0, len(weights)):
            nn = self.__hidden(weights[i], biases[i], x if nn is None else nn, name='layer' + str(i))

        # loss = tf.reduce_mean(tf.squared_difference(nn, y), name='loss')
        loss = tf.compat.v1.losses.huber_loss(y, nn)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss, name='train')

        # recommend = tf.math.subtract(nn, tf.math.ceil(y), 'recommend')
        recommend = nn

        return {
            'test': nn,
            'loss': loss,
            'train': train,
            'recommend': recommend,
            'placeholders': (x, y)
        }

    # Returns hidden layer made from inputs
    def __hidden(self, weights, biases, t_input, name):
        return tf.add(tf.matmul(t_input, weights), biases, name=name)

    def open_session(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def close_session(self):
        self.session.close()

    def train(self, data, labels, epochs, batch_size, print_interval=100):
        for epoch in range(1, epochs + 1):
            indices = range(epochs * batch_size, (epochs + 1) * batch_size)
            data_in = data.take(indices, axis=0, mode='wrap')
            labels_in = labels.take(indices, axis=0, mode='wrap')

            feed_dict = {
                self.model['placeholders'][0]: data_in,
                self.model['placeholders'][1]: labels_in
            }

            self.session.run([self.model['train']], feed_dict=feed_dict)

            if epoch % print_interval == 0:
                loss = self.session.run([self.model['loss']], feed_dict=feed_dict)
                print('Epoch ' + str(epoch) + ': loss=' + '{:.5f}'.format(loss[0]))

    def recommend(self, data):
        feed_dict = {
            self.model['placeholders'][0]: data,
            self.model['placeholders'][1]: data
        }
        result = self.session.run([self.model['recommend']], feed_dict=feed_dict)[0]
        return result
