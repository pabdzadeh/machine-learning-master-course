import glob
import os
import shutil

import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def prepare_data():
    paths = {'train': './data/seg_train/seg_train', 'test': './data/seg_test/seg_test'}
    parent_folders = {'train': os.listdir(paths['train']), 'test': os.listdir(paths['test'])}

    if not os.path.isdir('./combined_data'):
        os.mkdir('./combined_data')
    if not os.path.isdir('./combined_data/test'):
        os.mkdir('./combined_data/test')
    if not os.path.isdir('./combined_data/train'):
        os.mkdir('./combined_data/train')

    for mode in ['train', 'test']:
        parent_folder = parent_folders[mode]

        for folder in parent_folder:
            path_to_images = os.path.join(paths[mode], folder)
            images_path = glob.glob(os.path.join(path_to_images, '*.jpg'))

            path_to_train_combined_data = os.path.join('./combined_data/train/', folder)
            path_to_test_combined_data = os.path.join('./combined_data/test/', folder)

            x_train, x_test = train_test_split(images_path, test_size=0.3, shuffle=True)

            if not os.path.isdir(path_to_train_combined_data):
                os.mkdir(path_to_train_combined_data)
            if not os.path.isdir(path_to_test_combined_data):
                os.mkdir(path_to_test_combined_data)
            for data in x_train:
                shutil.copy(data, path_to_train_combined_data)
            for data in x_test:
                shutil.copy(data, path_to_test_combined_data)


def build_model(num_classes):
    model = tf.keras.Sequential([
        Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(150, 150, 3)),
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(.1),
        Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'),
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(.1),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'),
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(.1),
        Flatten(),
        Dense(units=32, activation='relu'),
        Dropout(.3),
        Dense(num_classes, activation='softmax')
    ])

    return model


# if it is first time using the dataset set this to False
is_data_prepared = False

if not is_data_prepared:
    print("Prepare and split data, start...")
    prepare_data()
    print("Prepare and split data, end.\n")

train_generator = ImageDataGenerator(
    rescale=1. / 255,
)

# if you want to test model with data augmentation set this parameter to True
augmentation = False

if augmentation:
    train_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

test_generator = ImageDataGenerator(
    rescale=1. / 255
)

batch_size = 64

train_data = train_generator.flow_from_directory(
    './combined_data/train/',
    class_mode='categorical',
    shuffle=True,
    target_size=(150, 150),
    color_mode='rgb',
    batch_size=batch_size
)

test_data = test_generator.flow_from_directory(
    './combined_data/test/',
    class_mode='categorical',
    shuffle=True,
    target_size=(150, 150),
    color_mode='rgb',
    batch_size=batch_size
)

train = True
test = True
epochs = 100
model = build_model(num_classes=6)

if train:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy', f1_m]
    )
    model.fit(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
    )
    model.summary()

if test:
    print('Evalution:', )
    model.evaluate(test_data)