import csv
import math
from collections import Counter


def dist(a, b):
    return math.sqrt(sum((0 if x == y else 1) for x, y in zip(a, b)))


def classify_knn(training_filename, testing_filename, k):
    with open(training_filename, "r") as f:
        reader = csv.reader(f)
        train_set = [(row[:-1], row[-1]) for row in reader]

    with open(testing_filename, "r") as f:
        reader = csv.reader(f)
        test_set = [row for row in reader]

    predictions = []

    for test_row in test_set:
        neighbours = sorted(train_set, key=lambda t: dist(test_row, t[0]))[:k]
        votes = Counter(label for _, label in neighbours)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            predictions.append("died")
        else:
            predictions.append(top[0][0])

    return predictions

def classify_knn_local(train_folds, test_fold, k):
    train_set = [(row[:-1], row[-1]) for row in train_folds]
    predictions = []

    for test_row in test_fold:
        test_attrs = test_row[:-1]
        neighbours = sorted(train_set, key=lambda t: dist(test_attrs, t[0]))[:k]
        votes = Counter(label for _, label in neighbours)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            predictions.append("died")
        else:
            predictions.append(top[0][0])

    return predictions