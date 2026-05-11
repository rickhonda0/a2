import csv

def read_data(data_filename):
    with open(data_filename, "r") as f:
        reader = csv.reader(f)
        data = [(row[:-1], row[-1]) for row in reader]

    return data

def cv(data_filename, n_folds):
    data = read_data(data_filename)
    n = len(data)

    # Group by class label
    groups = {}
    for instance in data:
        label = instance[1]
        if label not in groups:
            groups[label] = []
        
        instance[0].append(instance[1])
        groups[label].append(instance[0])

    # Split each class into k folds using divmod
    folds = [[] for _ in range(n_folds)]
    for label, instances in groups.items():
        subset_len, remainder = divmod(len(instances), n_folds)
        start = 0
        for i in range(n_folds):
            # stratification - foreach fold, give equal/similar proportions of the classes to the fold
            end = start + subset_len + (1 if i < remainder else 0)
            folds[i].extend(instances[start:end])
            start = end

    # Use each fold as test set, rest as training
    
    # with open("data/heart-folds.csv", "w") as f:
    #     for i in range(n_folds):
    #         f.write(f"fold{i+1}\n")
    #         for ex in folds[i]:
    #             f.write(",".join(ex))
    #             f.write("\n")

    #         f.write("\n")
    return folds, n
    

if __name__ == "__main__":
    cv("data/heart-failure-proc.csv", 10)