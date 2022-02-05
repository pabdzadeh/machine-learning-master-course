import PIL
from PIL import Image
import numpy as np

# define console colors
green = '\033[92m'
black = '\033[0m'


# knn implementation
def knn(train_set, train_labels, test_sample, k):
    dists_array = [np.linalg.norm(train_set[i] - test_sample) for i in range(len(train_set))]
    dists_array = np.array(dists_array)
    k_nearest_neighbours = dists_array.argsort()[:k]
    knn_labels = [train_labels[i] for i in k_nearest_neighbours]
    label = -1 if sum(knn_labels) < 0 else 1

    return label


# part (i) import dataset
print(f'{green}*** part (i): import dataset ***')
images_file = open('uspsdata.txt')
labels_file = open('uspscl.txt')

images_array = []
labels_array = []

for line in images_file:
    image_array = np.array(line.split())
    image_array = image_array.astype(float)
    images_array.append(image_array)

for line in labels_file:
    label_array = float(line)
    labels_array.append(label_array)

# part (ii) print four images
print(f'{green}*** part (ii): show first 4 images ***')
for i in range(4):
    image_array = images_array[i].reshape(16, 16)
    image = Image.fromarray(image_array)
    image.show()
    image = image.convert('L')
    image.save(str(i) + '.png')

# part (iii) split dataset to train, test, and validation
print(f'{green}*** part(iii): split dataset ***\n')
zipped_list = list(zip(labels_array, images_array))
np.random.seed(30)
np.random.shuffle(zipped_list)
labels_array, images_array = zip(*zipped_list)

train_set = images_array[:int(0.6 * len(images_array))]
validation_set = images_array[int(0.6 * len(images_array)): int(0.8 * len(images_array))]
test_set = images_array[int(0.8 * len(images_array)):]

train_labels = labels_array[:int(0.6 * len(images_array))]
validation_labels = labels_array[int(0.6 * len(images_array)): int(0.8 * len(images_array))]
test_labels = labels_array[int(0.8 * len(images_array)):]

# part (iv)
print(f'{green}*** part (iv): show test error for 1-nn ***{black}')
number_of_errors = 0
for i, item in enumerate(test_set):
    predicted_label = knn(train_set, train_labels, item, 1)

    if predicted_label != test_labels[i]:
        number_of_errors += 1
        image_array = item.reshape(16, 16)
        image = Image.fromarray(image_array)
        image = image.convert('L')
        image.save('misclassified_' + str(number_of_errors) + '.png')

print('test error:', number_of_errors / len(test_set), ' = ', 100*number_of_errors / len(test_set), '%')

# part (v)
print(f'{green}\n*** part (v): show test error for best k on validation ***{black}')
validation_errors = []
possible_k_values = [1, 3, 5, 7, 9, 11, 13]

for k in possible_k_values:
    number_of_errors = 0
    for i, item in enumerate(validation_set):
        predicted_label = knn(train_set, train_labels, item, k)

        if predicted_label != validation_labels[i]:
            number_of_errors += 1

    validation_errors.append(number_of_errors / len(validation_set))

print('Validation errors for [1, 3, 5, 7, 9, 11, 13]:', validation_errors)

validation_errors = np.array(validation_errors)
best_k = possible_k_values[validation_errors.argsort()[:1][0]]
print('Best k is the k with min validation error: ', best_k)

number_of_errors = 0
for i, item in enumerate(test_set):
    predicted_label = knn(train_set, train_labels, item, best_k)

    if predicted_label != test_labels[i]:
        number_of_errors += 1

print('test error:', number_of_errors / len(test_set))
