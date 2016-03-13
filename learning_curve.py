""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
# print data.DESCR


def show_samples():
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(np.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def teach_to_the_test(train_percentage):
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, train_size=train_percentage)
    model = LogisticRegression(C=10**-15)
    model.fit(X_train, y_train)
    # print "Train accuracy %f" % model.score(X_train, y_train)
    # print "Test accuracy %f" % model.score(X_test, y_test)
    return model.score(X_test, y_test)


def test_students(train_percentage, n=50):
    print 'testing robot students with {}% training'.format(
        int(train_percentage*100))
    accuracies = [teach_to_the_test(train_percentage) for _ in range(n)]
    return np.mean(accuracies)


num_trials = 10
train_percentages = np.arange(.05, .95, .05)
test_accuracies = [test_students(percent) for percent in train_percentages]

# print teach_to_the_test(0.9)

# train a model with training percentages between 5 and 90
# (see train_percentages) and evaluate the resultant accuracy.
# You should repeat each training percentage num_trials times
# to smooth out variability for consistency with the previous
# example use model = LogisticRegression(C=10**-10) for your learner

# TODO: your code here

fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()


