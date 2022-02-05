import numpy as np


class LogisticRegression:
    def __init__(self, batch_size=32, epochs=200, optimization='default'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimization = optimization
        self.weights = []

    def gradient(self, x, y, weights):
        y_hat = self.hypothesis(x, weights)
        g = x.transpose() @ (y_hat - y)
        return g

    def cost(self, x, y, weights):
        j = -y * np.log(self.hypothesis(x, weights)) - (1 - y) * np.log(1 - self.hypothesis(x, weights))
        return j.mean()

    def create_mini_batches(self, x, y, batch_size):
        mini_batches = []
        data = np.hstack((x, y))
        np.random.shuffle(data)
        n_mini_batches = data.shape[0] // batch_size
        i = 0

        for i in range(n_mini_batches + 1):
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
        return 1 / (1 + np.exp(-(x @ weights)) + 1e-8)

    def momuentum(self, v_t_minus_1, g_t, alpha=0.1, beta=0.999):
        v = beta * v_t_minus_1 + (1 - beta) * g_t
        step = alpha * v
        return step, v

    def train(self, x, y):
        w = np.zeros((x.shape[1], 1))
        error_list = []

        v_t = np.zeros((w.shape[0], 1))

        for p in range(self.epochs):
            mini_batches = self.create_mini_batches(x, y, self.batch_size)

            for mini_batch in mini_batches:
                x_mini, y_mini = mini_batch
                g_t = self.gradient(x_mini, y_mini, w)

                if self.optimization == 'momentum':
                    step, v_t = self.momuentum(v_t, g_t)
                else:
                    alpha_t = 1 / (p + 1)
                    step = alpha_t * g_t

                w = w - step
                loss = self.cost(x_mini, y_mini, w)

                print(f'epoch: {p}, loss: {loss}')

                error_list.append(loss)

                with open('logistic_' + str(self.optimization) + '_log.txt', 'a') as log_file:
                    log_file.write('epoch: ' + str(p) + ' loss:' + str(loss) + '\n')
        self.weights = w
        return w, error_list

    def predict(self, x):
        y_predicted = self.hypothesis(x, self.weights)
        y_predicted[np.where(y_predicted <= 0.5)] = 0
        y_predicted[np.where(y_predicted > 0.5)] = 1
        return y_predicted
