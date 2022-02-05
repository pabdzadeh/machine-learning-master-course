import numpy as np


class LinearRegression:
    def __init__(self, batch_size=32, epochs=100, optimization='default', write_loss_to_file=False, print_loss=False):
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimization = optimization
        self.weights = []
        self.name = 'linear'
        self.write_loss_to_file = write_loss_to_file
        self.print_loss = print_loss

    def gradient(self, x, y, weights):
        h = self.hypothesis(x, weights)
        g = x.transpose() @ (h - y)
        return g

    def cost(self, x, y, weights):
        y_hat = self.hypothesis(x, weights)
        j = (y_hat - y).transpose() @ (y_hat - y)
        j /= 2
        return j[0]

    def create_mini_batches(self, x, y, batch_size):
        mini_batches = []
        data = np.hstack((x, y))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // batch_size
        i = 0

        for i in range(n_minibatches + 1):
            mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
            x_mini = mini_batch[:, :-1]
            y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((x_mini, y_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            x_mini = mini_batch[:, :-1]
            y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((x_mini, y_mini))
        return mini_batches

    def hypothesis(self, x, weights):
        return x @ weights

    def adagrad(self, g_t, s_t_minus_1, epsilon=1e-6, alpha=1):
        s_t = s_t_minus_1 + g_t ** 2
        step = (alpha / np.sqrt(s_t + epsilon)) * g_t
        return step, s_t

    def momuentum(self, v_t_minus_1, g_t, alpha=0.01, beta=0.99):
        v = beta * v_t_minus_1 + (1 - beta) * g_t
        step = alpha * v
        return step, v

    def train(self, x, y):
        w = np.zeros((x.shape[1], 1))
        error_list = []
        g_t_minus_1_bar = 1

        s_t = np.zeros((x.shape[1], 1))
        v_t = np.zeros((w.shape[0], 1))

        for p in range(self.epochs):
            mini_batches = self.create_mini_batches(x, y, self.batch_size)

            for mini_batch in mini_batches:
                x_mini, y_mini = mini_batch
                g_t = self.gradient(x_mini, y_mini, w)

                if self.optimization == 'adagrad':
                    step, s_t = self.adagrad(g_t, s_t)

                elif self.optimization == 'momentum':
                    step, v_t = self.momuentum(v_t, g_t)

                else:
                    g_t_bar = g_t_minus_1_bar + 1 / (g_t.shape[0] + 1) * np.sum(np.abs(g_t))
                    g_t_minus_1_bar = g_t_bar
                    alpha_t = 1 / (1 + g_t_bar)
                    step = alpha_t * g_t
                w = w - step

                loss = self.cost(x_mini, y_mini, w)[0]
                error_list.append(loss)

                if self.write_loss_to_file:
                    with open(self.name + '_' + str(self.optimization) + '_log.txt', 'a') as log_file:
                        log_file.write('run: ' + str(self.write_loss_to_file) + ' epoch: ' + str(p) + ' loss:' + str(loss) + '\n')
                if self.print_loss:
                    print(f'run:{int(self.write_loss_to_file)} epoch: {p}, loss: {loss}')

        self.weights = w

        return w, error_list

    def predict(self, x_test):
        y_predicted = self.hypothesis(x_test, self.weights)
        return y_predicted
