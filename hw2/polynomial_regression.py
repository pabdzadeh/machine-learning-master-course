from linear_regression import LinearRegression
import numpy as np


class PolynomialRegression(LinearRegression):
    def __init__(self, batch_size=32, epochs=100, optimization='default', write_loss=False):
        super().__init__(batch_size, epochs, optimization, write_loss_to_file=write_loss)
        self.name = 'poly'

    def transform(self, x):
        x_pow = np.power(x[:, 1:], 2).reshape(-1, 7)
        x_transform = np.append(x, x_pow, axis=1)
        return x_transform
