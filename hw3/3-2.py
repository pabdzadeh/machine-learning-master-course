import numpy as np
import csv


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def estimate_parameters_with_MLE(x_train, y_train):
    N_j1 = np.zeros(x_train.shape[1])
    N_j0 = np.zeros(x_train.shape[1])
    N1 = y_train.sum()
    N0 = y_train.shape[0] - N1

    theta_c0 = N0 / y_train.shape[0]
    theta_c1 = N1 / y_train.shape[0]

    for i, item in enumerate(x_train):
        for j, feature in enumerate(item):

            if y_train[i] == 0 and feature == 1:
                N_j0[j] += 1
            if y_train[i] == 1 and feature == 1:
                N_j1[j] += 1

    theta_features_c0 = N_j0 / N0
    theta_features_c1 = N_j1 / N1

    return theta_c0, theta_c1, theta_features_c0, theta_features_c1


def estimate_parameters_with_MAP(x_train, y_train):
    N_j0 = np.zeros(x_train.shape[1])
    N_j1 = np.zeros(x_train.shape[1])

    N1 = y_train.sum()
    N0 = y_train.shape[0] - N1

    theta_c0 = (N0 + 1) / (y_train.shape[0] + 1)
    theta_c1 = (N1 + 1) / (y_train.shape[0] + 1)

    for i, item in enumerate(x_train):
        for j, feature in enumerate(item):
            if y_train[i] == 0 and feature == 1:
                N_j0[j] += 1

            if y_train[i] == 1 and feature == 1:
                N_j1[j] += 1

    theta_features_c1 = (N_j1 + 1) / (N1 + 2)
    theta_features_c0 = (N_j0 + 1) / (N0 + 2)

    return theta_c0, theta_c1, theta_features_c0, theta_features_c1


def predict_based_on_MLE(x_test_item, theta_c0, theta_c1, theta_features_c0,
                         theta_features_c1):
    p_y0 = theta_c0
    p_y1 = theta_c1

    for i, col in enumerate(x_test_item):
        if col == 0:
            p_y0 *= (1 - theta_features_c0[i])
            p_y1 *= (1 - theta_features_c1[i])
        else:
            p_y0 *= theta_features_c0[i]
            p_y1 *= theta_features_c1[i]

    return 1 if p_y1 >= p_y0 else 0


def predict_based_on_MAP(x_test_item, theta_c0, theta_c1, theta_features_c0,
                         theta_features_c1):
    p_y0 = theta_c0
    p_y1 = theta_c1

    for i, col in enumerate(x_test_item):
        if col == 0:
            p_y0 *= (1 - theta_features_c0[i])
            p_y1 *= (1 - theta_features_c1[i])
        else:
            p_y0 *= theta_features_c0[i]
            p_y1 *= theta_features_c1[i]

    return 1 if p_y1 >= p_y0 else 0


def print_specifics(type, theta_c0, theta_c1, theta_features_c0, theta_features_c1, round=3):
    print(f'Specs of run={run + 1} and {type} ---->')
    print(f'Theta C for C = 1: {np.round(theta_c1, round)}')
    print(f'Theta C for C = 0: {np.round(theta_c0, round)}')
    print(f'Theta for features given c = 1 : {np.round(theta_features_c1, round)}')
    print(f'Theta for features given c = 0 : {np.round(theta_features_c0, round)}')
    if type == 'MLE':
        print('.....................................................................')
    else:
        print('')


rows = []

with open('breast.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    fields = next(csvreader)

    for row in csvreader:
        rows.append(row)

rows = np.array(rows, dtype=int)

where_are_2s = np.where(rows == 2)
rows[where_are_2s] = 0
where_are_ups = np.where(rows == 4)
rows[where_are_ups] = 1

misclassification_rates = []
misclassification_rates_MAP = []

for run in range(20):
    np.random.shuffle(rows)
    x_train = rows[:int(0.8 * rows.shape[0]), :-1]
    y_train = rows[:int(0.8 * rows.shape[0]), -1:].reshape(-1, 1)
    x_test = rows[int(0.8 * rows.shape[0]):, :-1]
    y_test = rows[int(0.8 * rows.shape[0]):, -1:].reshape(-1, 1)

    theta_c0, theta_c1, theta_features_c0, theta_features_c1 = estimate_parameters_with_MLE(x_train, y_train)

    print(f'{Colors.WARNING}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{Colors.ENDC}')
    print_specifics('MLE', theta_c0, theta_c1, theta_features_c0, theta_features_c1)

    misclassifications = 0

    for i, item in enumerate(x_test):
        predicted_class = predict_based_on_MLE(item, theta_c0, theta_c1, theta_features_c0, theta_features_c1)

        if predicted_class != y_test[i]:
            misclassifications += 1

    misclassification_rates.append(misclassifications / y_test.shape[0])

    misclassifications_MAP = 0

    theta_c0, theta_c1, theta_features_c0, theta_features_c1 = estimate_parameters_with_MAP(x_train, y_train)

    print_specifics('MAP', theta_c0, theta_c1, theta_features_c0, theta_features_c1)

    for i, item in enumerate(x_test):
        predicted_class = predict_based_on_MAP(item, theta_c0, theta_c1, theta_features_c0, theta_features_c1)
        if predicted_class != y_test[i]:
            misclassifications_MAP += 1

    misclassification_rates_MAP.append(misclassifications_MAP / y_test.shape[0])

    print(f'{Colors.OKGREEN}Run={run + 1}, classification error rate MLE: '
          f'{round(misclassifications / y_test.shape[0] * 100, 2)}%, MAP: '
          f'{round(misclassifications_MAP / y_test.shape[0] * 100, 2)}%{Colors.ENDC}')
    print(f'{Colors.WARNING}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{Colors.ENDC}\n')

misclassification_rates = np.array(misclassification_rates)
misclassification_rates_MAP = np.array(misclassification_rates_MAP)

print(f'{Colors.OKBLUE}\n***************************************')
print('MLE mean classification error: ', round(misclassification_rates.mean() * 100, 3), '%')
print('MAP mean classification error: ', round(misclassification_rates_MAP.mean() * 100, 3), '%')
print('***************************************')
