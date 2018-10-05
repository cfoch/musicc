import argparse
import math
import numpy as np
import pickle
import sys

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


def flattenize(data, feature):
    x = []
    y = []
    for genre in data:
        for file_path in data[genre]:
            feature_data = data[genre][file_path][feature]
            x.append(feature_data)
            y.append(genre)
    return x, y

def randomize_data(data):
    data = list(zip(*data))
    shuffle(data)
    data = list(zip(*data))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    parser.add_argument("-f", "--feature",
                        help="The feature to use",
                        required=True)
    parser.add_argument("-l", "--list-features", action="store_true",
                        help="List the available features",
                        required=False)
    """
    parser.add_argument("-i", "--input",
                        type=argparse.FileType('rb'),
                        help="Path to pickle generated by musicc-datagen.",
                        required=True)
    parser.add_argument("-o", "--output",
                        type=argparse.FileType('wb'),
                        help="Path to output pickle containing x (features) "
                        "and y (labels)",
                        required=False)
    parser.add_argument("--batch-size",
                        type=int,
                        help="Batch Size",
                        required=False)
    parser.add_argument("--standarize", action="store_true",
                        help="Whether standarize the data.",
                        required=False)

    args = parser.parse_args()
    data = pickle.load(args.input)
    x_train, x_test, y_train, y_test =\
        train_test_split(data[0], data[1], test_size=0.33, random_state=42)

    if args.standarize:
        x_train = (x_train - x_train.mean()) / x_train.std()
        x_test = (x_train - x_test.mean()) / x_test.std()

    x_train = [matrix[:,:,None] for matrix in x_train]
    x_test = [matrix[:,:,None] for matrix in x_test]

    lr = 0.01
    bs = args.batch_size or 50
    nb = math.ceil(len(x_train) / bs)

    model = Sequential([
        Conv2D(32, 3, activation='relu', padding='same', input_shape=x_train[0].shape),
        MaxPool2D(),
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPool2D(),
        Conv2D(128, 3, activation='relu', padding='same'),
        MaxPool2D(),
        Flatten(),
        Dense(10, activation='softmax')
    ])

    model.compile(SGD(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    log = model.fit(x_train, y_train, batch_size=bs, epochs=6, validation_data=[x_test, y_test])