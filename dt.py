import csv
import math


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


def classify_dt(training_filename, testing_filename):
    with open(training_filename, "r") as f:
        reader = csv.reader(f)
        train_set = [(row[:-1], row[-1]) for row in reader]

    with open(testing_filename, "r") as f:
        reader = csv.reader(f)
        test_set = [row for row in reader]

    attr_indices = list(range(len(train_set[0][0])))
    
    # Training
    tree = build_tree(train_set, attr_indices, train_set)

    # Inference
    results = []
    for instance in test_set:
        results.append(classify_one(tree, instance))
    return results