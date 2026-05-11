import csv
import math
import random
from collections import Counter


def entropy(examples):
    if not examples:
        return 0.0
    total = len(examples)
    counts = {}
    for _, label in examples:
        counts[label] = counts.get(label, 0) + 1
    ent = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def majority_class(examples):
    counts = {}
    for _, label in examples:
        counts[label] = counts.get(label, 0) + 1
    best = max(counts, key=lambda c: (counts[c], c == "died"))
    return best


def info_gain(examples, attr_idx):
    total = len(examples)
    partitions = {}
    for attrs, label in examples:
        val = attrs[attr_idx]
        if val not in partitions:
            partitions[val] = []
        partitions[val].append((attrs, label))
    remainder = sum(
        (len(subset) / total) * entropy(subset)
        for subset in partitions.values()
    )
    return entropy(examples) - remainder


def build_tree(examples, attr_indices, parent_examples):
    if not examples:
        return majority_class(parent_examples)

    labels = set(label for _, label in examples)
    if len(labels) == 1:
        return examples[0][1]

    if not attr_indices:
        return majority_class(examples)

    best_attr = max(attr_indices, key=lambda i: info_gain(examples, i))

    partitions = {}
    for attrs, label in examples:
        val = attrs[best_attr]
        if val not in partitions:
            partitions[val] = []
        partitions[val].append((attrs, label))

    remaining = [i for i in attr_indices if i != best_attr]
    default = majority_class(examples)
    branches = {}
    for val, subset in partitions.items():
        branches[val] = build_tree(subset, remaining, examples)
    return (best_attr, branches, default)


def classify_one(tree, instance):
    if isinstance(tree, str):
        return tree
    attr_idx, branches, default = tree
    val = instance[attr_idx]
    if val in branches:
        return classify_one(branches[val], instance)
    return default


def print_tree(tree, indent=""):
    if isinstance(tree, str):
        print(f"{indent}-> {tree}")
        return
    attr_idx, branches, default = tree
    for val, subtree in sorted(branches.items()):
        print(f"{indent}[Attr {attr_idx} = {val}]")
        print_tree(subtree, indent + "  ")


# KNN+: Information-gain-weighted distance + majority voting

def classify_knn_plus(training_filename, testing_filename, k):
    with open(training_filename, "r") as f:
        train_set = [(row[:-1], row[-1]) for row in csv.reader(f)]

    with open(testing_filename, "r") as f:
        test_set = [row for row in csv.reader(f)]

    n_attrs = len(train_set[0][0])

    # Weight each attribute by its information gain
    attr_weights = [info_gain(train_set, i) for i in range(n_attrs)]
    total_w = sum(attr_weights)
    if total_w > 0:
        attr_weights = [w / total_w for w in attr_weights]

    def wdist(a, b):
        return sum(attr_weights[i] for i in range(n_attrs) if a[i] != b[i])

    results = []
    for instance in test_set:
        test_attrs = instance[:n_attrs]
        neighbours = sorted(train_set, key=lambda t: wdist(test_attrs, t[0]))[:k]
        votes = Counter(label for _, label in neighbours)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            results.append("died")
        else:
            results.append(top[0][0])
    return results


def classify_knn_plus_local(train_folds, test_fold, k):
    train_set = [(row[:-1], row[-1]) for row in train_folds]

    n_attrs = len(train_set[0][0])
    attr_weights = [info_gain(train_set, i) for i in range(n_attrs)]
    total_w = sum(attr_weights)
    if total_w > 0:
        attr_weights = [w / total_w for w in attr_weights]

    def wdist(a, b):
        return sum(attr_weights[i] for i in range(n_attrs) if a[i] != b[i])

    results = []
    for instance in test_fold:
        test_attrs = instance[:-1]
        neighbours = sorted(train_set, key=lambda t: wdist(test_attrs, t[0]))[:k]
        votes = Counter(label for _, label in neighbours)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            results.append("died")
        else:
            results.append(top[0][0])
    return results


# DT+: Gain Ratio (C4.5) + Reduced Error Pruning

def gain_ratio(examples, attr_idx):
    ig = info_gain(examples, attr_idx)
    total = len(examples)
    value_counts = {}
    for attrs, _ in examples:
        val = attrs[attr_idx]
        value_counts[val] = value_counts.get(val, 0) + 1
    split_info = 0.0
    for count in value_counts.values():
        p = count / total
        if p > 0:
            split_info -= p * math.log2(p)
    if split_info == 0:
        return 0.0
    return ig / split_info


def build_tree_plus(examples, attr_indices, parent_examples):
    if not examples:
        return majority_class(parent_examples)

    labels = set(label for _, label in examples)
    if len(labels) == 1:
        return examples[0][1]

    if not attr_indices:
        return majority_class(examples)

    # C4.5 heuristic: only consider attributes with above-average IG,
    # then pick the one with the best gain ratio among those
    gains = {i: info_gain(examples, i) for i in attr_indices}
    avg_gain = sum(gains.values()) / len(gains)
    candidates = [i for i in attr_indices if gains[i] >= avg_gain]
    if not candidates:
        candidates = list(attr_indices)

    best_attr = max(candidates, key=lambda i: gain_ratio(examples, i))

    partitions = {}
    for attrs, label in examples:
        val = attrs[best_attr]
        if val not in partitions:
            partitions[val] = []
        partitions[val].append((attrs, label))

    remaining = [i for i in attr_indices if i != best_attr]
    default = majority_class(examples)
    branches = {}
    for val, subset in partitions.items():
        branches[val] = build_tree_plus(subset, remaining, examples)
    return (best_attr, branches, default)


def prune_tree(tree, val_examples):
    if isinstance(tree, str):
        return tree

    attr_idx, branches, default = tree

    new_branches = {}
    for val, subtree in branches.items():
        child_examples = [(a, l) for a, l in val_examples if a[attr_idx] == val]
        new_branches[val] = prune_tree(subtree, child_examples)

    pruned = (attr_idx, new_branches, default)

    tree_correct = sum(1 for a, l in val_examples if classify_one(pruned, a) == l)
    leaf_correct = sum(1 for _, l in val_examples if default == l)

    if leaf_correct >= tree_correct:
        return default
    return pruned


def classify_dt_plus(training_filename, testing_filename):
    with open(training_filename, "r") as f:
        data = [(row[:-1], row[-1]) for row in csv.reader(f)]

    with open(testing_filename, "r") as f:
        test_set = [row for row in csv.reader(f)]

    # Stratified 80/20 split for pruning validation
    rng = random.Random(42)
    died = [x for x in data if x[1] == "died"]
    survived = [x for x in data if x[1] == "survived"]
    rng.shuffle(died)
    rng.shuffle(survived)

    split_d = int(0.8 * len(died))
    split_s = int(0.8 * len(survived))
    train = died[:split_d] + survived[:split_s]
    val = died[split_d:] + survived[split_s:]

    attr_indices = list(range(len(data[0][0])))
    tree = build_tree_plus(train, attr_indices, train)
    tree = prune_tree(tree, val)

    results = []
    for instance in test_set:
        results.append(classify_one(tree, instance))
    return results


def classify_dt_plus_local(train_folds, test_fold):
    data = [(row[:-1], row[-1]) for row in train_folds]

    rng = random.Random(42)
    died = [x for x in data if x[1] == "died"]
    survived = [x for x in data if x[1] == "survived"]
    rng.shuffle(died)
    rng.shuffle(survived)

    split_d = int(0.8 * len(died))
    split_s = int(0.8 * len(survived))
    train = died[:split_d] + survived[:split_s]
    val = died[split_d:] + survived[split_s:]

    attr_indices = list(range(len(data[0][0])))
    tree = build_tree_plus(train, attr_indices, train)
    tree = prune_tree(tree, val)

    return [classify_one(tree, instance[:-1]) for instance in test_fold]
