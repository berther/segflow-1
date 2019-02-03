import argparse
import os
from utils.io import read_image
import numpy as np
from utils.labels import label_to_mask, label_to_categories
import json
from typing import List


def main():
    parser = argparse.ArgumentParser("class weight computation")
    parser.add_argument(
        "--train_label_dir",
        default="datasets/city_full/labels_ids/train")
    parser.add_argument(
        "--lim", type=int, default=-1, help="limit label count for testing")
    parser.add_argument(
        "--n_labels", type=int, default=20, help="how many label classes")
    parser.add_argument(
        "--n_categories", type=int, default=8, help="how many categories")
    parser.add_argument(
        "--ignore_classes", nargs="*", type=int, default=[19],
        help="which classes to always give zero weight")
    args = parser.parse_args()

    # find unique classes
    uniques, uniques_cat = [], []
    # count classes and categories over each pixel of every image
    class_counts = np.zeros(args.n_labels)
    cat_counts = np.zeros(args.n_categories)
    ctotal = 0
    done = False
    for root, dirs, files in os.walk(args.train_label_dir):
        if done:
            break
        root = root.replace("\\", "/")
        for i, f in enumerate(files):
            # progress output and canceling during small test
            if i % 200 == 0:
                print("{}/{} {}".format(i, len(files), root))
            if -1 < args.lim < ctotal:
                print("label count limit --lim reached!")
                done = True
                break
            # load label
            file_full = root + "/" + f
            label = read_image(file_full, True)
            # count train labels
            mask = label_to_mask(label)
            unique, counts = np.unique(mask, return_counts=True)
            uniques.extend(unique)
            uniques = list(np.unique(uniques))
            class_counts[unique] += counts
            # count categories
            cats = label_to_categories(label)
            unique, counts = np.unique(cats, return_counts=True)
            uniques_cat.extend(unique)
            uniques_cat = list(np.unique(uniques_cat))
            cat_counts[unique] += counts

            ctotal += 1
    print()
    json_dir = "results_json"
    os.makedirs(json_dir, exist_ok=True)

    # ----- classes
    print("----- classes (fine labels)")
    weights = compute_weights(
        uniques, class_counts, args.n_labels, empty_labels=args.ignore_classes,
        verbose=True)
    json_file = json_dir + "/weights_labels.json"
    with open(json_file, "wt") as fh:
        json.dump(list(weights), fh)
    print("wrote class weights to", json_file)
    print()

    # ----- categories
    print("----- categories (coarse labels)")
    weights = compute_weights(
        uniques_cat, cat_counts, args.n_categories, empty_labels=[],
        verbose=True)
    json_file = json_dir + "/weights_categories.json"
    with open(json_file, "wt") as fh:
        json.dump(list(weights), fh)
    print("wrote category weights to", json_file)


def compute_weights(
        uniques, class_counts, n_labels, empty_labels: List = [],
        verbose=False):
    """Balanced weight mask over every pixel's class of each label
    (classes that appear less often have bigger weight)
    """
    # some specified classes should not get any weights, so set the count to 0
    for l in empty_labels:
        class_counts[l] = 0
    # weight is then total sum over count per class times total num classes
    sum_total = np.sum(class_counts)
    weights = sum_total / class_counts
    # all count zeros have infinite weight, set to zero
    weights[np.isinf(weights)] = 0
    # get effective number of classes
    num_zeroes = np.sum(weights == 0)
    eff_n_classes = n_labels - num_zeroes
    # scale so weights sum up to number of effective classes
    # # weights = weights / (n_labels - 2) * (args.n_labels - 1)
    weights = weights / np.sum(weights) * eff_n_classes
    if verbose:
        print("uniques", uniques)
        print("counts:", class_counts)
        print("weights", weights)
        print("sum of weights", np.sum(weights))
        print("num_zeroes", num_zeroes, "eff_classes", eff_n_classes)
    return weights


if __name__ == '__main__':
    main()
