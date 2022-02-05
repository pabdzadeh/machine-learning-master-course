import numpy as np
import matplotlib.pyplot as plt
import csv

from knn import knn
from logistic_regression import LogisticRegression


def normalize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


fields = []
rows = []

with open('Weekly.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)

    for row in csvreader:
        rows.append(row)


rows = np.array(rows)
rows = rows[:, 1:]

where_are_downs = np.where(rows == 'Down')
rows[where_are_downs] = 0
where_are_ups = np.where(rows == 'Up')
rows[where_are_ups] = 1

rows = np.array(rows, dtype=float)
np.random.shuffle(rows)


x_train = rows[:int(0.8 * rows.shape[0]), :-1]
y_train = rows[:int(0.8 * rows.shape[0]), -1:].reshape(-1, 1)
x_test = rows[int(0.8 * rows.shape[0]):, :-1]
y_test = rows[int(0.8 * rows.shape[0]):, -1:].reshape(-1, 1)

x_train = normalize(x_train)
x_test = normalize(x_test)

x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

for optimization in ['default', 'momentum']:
    # optimization = 'default'
    model = LogisticRegression(optimization=optimization)
    weights, loss_list = model.train(x_test, y_test)
    y_predicted = model.predict(x_test)

    error = np.sum(np.abs(y_test - y_predicted)) / y_test.shape[0]

    print('\n**********************************')
    print(f'logistic regression {optimization}, error: {error}, classification accuracy: {(1-error) * 100}%,'
          f' total number of test items: {x_test.shape[0]}')
    print('**********************************\n')


x_train = rows[:int(0.8 * rows.shape[0]), :-1]
y_train = rows[:int(0.8 * rows.shape[0]), -1:].reshape(-1, 1)
x_test = rows[int(0.8 * rows.shape[0]):, :-1]
y_test = rows[int(0.8 * rows.shape[0]):, -1:].reshape(-1, 1)

print('classification error for knn:')
for k in [3, 5, 7]:
    number_of_errors = 0
    for i, item in enumerate(x_test):
        predicted_label = knn(x_train, y_train, item, k)

        if predicted_label != y_test[i]:
            number_of_errors += 1

    print(f'test error for k={k}: ', number_of_errors / y_test.shape[0])