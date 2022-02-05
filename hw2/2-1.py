import numpy as np
import csv

from linear_regression import LinearRegression
from polynomial_regression import PolynomialRegression


def normalize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

rows = []

with open('Auto.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)

    for row in csvreader:
        rows.append(row)

rows = np.array(rows)
rows = rows[:, :-1]
where_are_NaNs = np.where(rows == '?')
rows = np.delete(rows, where_are_NaNs[0], axis=0)
rows = np.array(rows, dtype=float)
indices = [1, 2, 3, 4, 5, 6, 7, 0]
rows = rows[:, indices]


errors = []
mode = 'linear_regression'
optimization = 'adagrad'
# optimization = 'default'
# optimization = 'momentum'

print(f'{mode} opt={optimization}')
print('---------------------------------------------')
for run in range(100):
    np.random.shuffle(rows)

    x_train = rows[:int(0.8 * rows.shape[0]), :-1]
    y_train = rows[:int(0.8 * rows.shape[0]), -1:].reshape(-1, 1)
    x_test = rows[int(0.8 * rows.shape[0]):, :-1]
    y_test = rows[int(0.8 * rows.shape[0]):, -1:].reshape(-1, 1)
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

    if mode == 'linear_regression':
        model = LinearRegression(optimization=optimization, write_loss_to_file=(run if run == 10 else False))
        weights, loss_list = model.train(x_train, y_train)
        y_predicted = model.predict(x_test)

    elif mode == 'polynomial_regression':
        model = PolynomialRegression(write_loss=(run if run == 10 else False))
        x_train = model.transform(x_train)
        weights, loss_list = model.train(x_train, y_train)
        x_test = model.transform(x_test)
        y_predicted = model.predict(x_test)

    error = np.sum(np.abs(y_test - y_predicted)) / y_test.shape[0]
    print(f'|||| run:{run}, error: {error} ||||')
    print('---------------------------------------------')
    errors.append(error)

print('\n**************************************************************************')
print(f'{mode} optimization={optimization} mean error: {np.mean(np.array(errors))}')
print('**************************************************************************\n\n')

# compare momentum with default optimization
see_one_epoch_of_each_for_comparison = False # change to true to compare momentum and default

if see_one_epoch_of_each_for_comparison:
    print('******************************************')
    print('compare momentum with default optimization')
    print('******************************************')

    model = LinearRegression(optimization='default', print_loss=True)
    weights, loss_list = model.train(x_train, y_train)
    y_predicted = model.predict(x_test)
    error = np.sum(np.abs(y_test - y_predicted)) / y_test.shape[0]
    print(f'default, error: {error}, min loss: {min(loss_list)}, min loss index: {loss_list.index(min(loss_list))}')
    print('******************************************')

    model = LinearRegression(optimization='momentum', print_loss=True)
    weights, loss_list = model.train(x_train, y_train)
    y_predicted = model.predict(x_test)
    error = np.sum(np.abs(y_test - y_predicted)) / y_test.shape[0]
    print(f'momentum, error: {error}, min loss: {min(loss_list)}, min loss index: {loss_list.index(min(loss_list))}')
