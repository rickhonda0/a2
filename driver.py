import sys
import time
import math
from datetime import datetime as dt
from pathlib import Path
from matplotlib import pyplot as plt
from cross_val import cv
from dt import classify_dt_local
from knn import classify_knn_local
from knn_dt_plus import classify_knn_plus_local, classify_dt_plus_local

CLASSES = [
    "died",
    "survived"
]

def compare(classified_A, classified_B):
    """perform 2-tailed t-test"""

def classify(classifier, classifier_name, data_fname, n_folds=10, k=10):
    folds, n_datapoints = cv(data_fname, n_folds)

    tps, tns, fps, fns, runtimes = [], [], [], [], []

    for i in range(n_folds):
        X_test = folds[i]
        X_train = [ex for fold in (folds[:i] + folds[(i+1):]) for ex in fold]

        if classifier_name in ("knn", "knn+"):
            start = time.perf_counter()
            class_preds = classifier(X_train, X_test, k)
            end = time.perf_counter()

        else:
            start = time.perf_counter()
            class_preds = classifier(X_train, X_test)
            end = time.perf_counter()

        runtimes.append(end - start)

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for j in range(len(X_test)):
            y = X_test[j][-1]
            y_hat = class_preds[j]
            match y:
                # positive := died
                # negative := survived
                case "died":
                    if y == y_hat:
                        tp += 1
                    elif y_hat == "survived":
                        fn += 1
                case "survived":
                    if y == y_hat:
                        tn += 1
                    elif y_hat == "died":
                        fp += 1

        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)

    def mean(xs): return sum(xs) / len(xs)
    def stdev(xs):
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    accs = [(tp+tn)/(tp+tn+fp+fn) for tp, tn, fp, fn in zip(tps, tns, fps, fns)]
    precisions = [tp/(tp+fp) if (tp+fp) > 0 else 0.0 for tp, fp in zip(tps, fps)]
    recalls = [tp/(tp+fn) if (tp+fn) > 0 else 0.0 for tp, fn in zip(tps, fns)]
    f1s = [2*p*r/(p+r) if (p+r) > 0 else 0.0 for p, r in zip(precisions, recalls)]

    mean_acc, std_acc = mean(accs), stdev(accs)
    mean_prec, std_prec = mean(precisions), stdev(precisions)
    mean_rec, std_rec = mean(recalls), stdev(recalls)
    mean_f1, std_f1 = mean(f1s), stdev(f1s)
    mean_rt, std_rt = mean(runtimes), stdev(runtimes)

    summary = (
        f"=== {classifier_name} ({n_folds}-fold CV) ===\n"
        f"n:         {n_datapoints}\n"
        f"Accuracy:  {mean_acc:.4f} +/- {std_acc:.4f}\n"
        f"Precision: {mean_prec:.4f} +/- {std_prec:.4f}\n"
        f"Recall:    {mean_rec:.4f} +/- {std_rec:.4f}\n"
        f"F1:        {mean_f1:.4f} +/- {std_f1:.4f}\n"
        f"Average elapsed time: {mean_rt:.4f} +/- {std_rt:.4f}\n"
    )

    now = dt.now()
    run_dir = Path("logs") / f"{now.strftime('%m-%d_%H-%M')}_{classifier_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "summary.txt", "w") as f:
        f.write(summary)

    total_tp, total_tn = sum(tps), sum(tns)
    total_fp, total_fn = sum(fps), sum(fns)

    confusion_mtrx = [
        [total_tp, total_fn],
        [total_fp, total_tn]]

    fig, ax = plt.subplots()
    ax.imshow(confusion_mtrx, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, confusion_mtrx[i][j], ha="center", va="center", fontsize=16)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{classifier_name} Confusion Matrix")

    plt.tight_layout()
    plt.savefig(run_dir / "confusion.png", dpi=150)
    plt.close()


        
        

CLASSIFIERS = {
    "dt":   classify_dt_local,
    "dt+":  classify_dt_plus_local,
    "knn":  classify_knn_local,
    "knn+": classify_knn_plus_local,
    "all":  None,
}

if __name__ == "__main__":
    DATA = "data/heart-failure-proc.csv"

    choices = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    if "all" in choices:
        choices = [k for k in CLASSIFIERS if k != "all"]

    for name in choices:
        if name not in CLASSIFIERS:
            print(f"Unknown classifier: {name}")
            print(f"Options: {', '.join(CLASSIFIERS)}")
            sys.exit(1)
        print(f"Running {name}...")
        classify(CLASSIFIERS[name], name, DATA)