# coding=utf-8
# Your name :Valinda Vanam
# Certificate of Authenticity: “I certify that the code in the method functions including method function main of this project is entirely my own work”

import sys
from itertools import repeat
import numpy as np

#Getting input from user for degrees in int data type
degree = int(sys.argv[-1])

#lr-Learning rate - learning rate can be randomely given
lr = 0.005

#epochs are the iterations
epochs = 10000

#threshold value also can be adjusted randomly for the given data set
threshold = 4

# array to store the training data set
x_train = []
y_train = []

# loading training files and save the content in two arrays
dataset = open(sys.argv[-3], 'r')
data = dataset.read().splitlines()
for d in data:
    int_data = list(map(int, d.split()))
    y_train.append(int_data.pop())
    x_train.append(int_data)

# array to store the testing dataset
x_test = []
y_test = []

dataset = open(sys.argv[-2], 'r') #opening file in read mode and assigning to a dataset
data = dataset.read().splitlines()
for d in data:
    int_data = list(map(int, d.split()))
    y_test.append(int_data.pop())
    x_test.append(int_data)

print(f"The length of training dataset and testing dataset is {len(x_train)} and {len(x_test)}")


def normalize(column):
    return (column - min(column)) / (max(column) - min(column))


x_trainNormalised = np.zeros(np.array(x_train).shape)
x_testNormalised = np.zeros(np.array(x_test).shape)

# preprocessing both the array using min-max normalization
for i, col in enumerate(zip(np.array(x_train).T, np.array(x_test).T)):
    x_trainNormalised[:, i] = normalize(col[0])
    x_testNormalised[:, i] = normalize(col[1])

print(
    f"The shape of normalized training dataset and testing dataset is {x_trainNormalised.shape} and {x_testNormalised.shape}")

theta = np.zeros(x_trainNormalised.shape[1])
error = []
cnt = 0
i = 0

#Linear regression algorithm implementation
while i <= epochs and cnt < 3:
    err = x_trainNormalised.dot(theta) - y_train
    theta -= lr * (x_trainNormalised.T.dot(err)) / len(y_train)
    temp = np.sum((x_trainNormalised.dot(theta) - y_train) ** 2) / (2 * len(y_train))
    error.append(temp)
    if temp <= threshold:
        cnt += 1
    i += 1

print("The outputs of the training phase are")
for i, th in enumerate(theta):
    print(f"\u03F4{i + 1} = %0.4f" % th)


def predict(degree, theta, xtest):
    temp = [test ** i for test in xtest for i in range(1, degree + 1)]
    temp.insert(0, 1)
    return sum(th * da for th, da in zip(theta, temp))


print()
predictedOutput = list(map(predict, repeat(degree), repeat(theta), x_testNormalised))
predictedError = [(i - j) ** 2 for i, j in zip(y_test, predictedOutput)]
print('Object ID', 'output', 'target value', 'squared error', sep="\t")
for i, data in enumerate(zip(y_test, predictedOutput, predictedError)):
    print("%05d" % i, "%14.4f" % data[0], "%10.4f" % data[1], "%0.4f" % data[2], sep="\t")
