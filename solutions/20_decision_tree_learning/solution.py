import math
from collections import Counter
import numpy as np

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    labels = np.array(labels)
    unique_values, counts = np.unique(labels, return_counts=True)
    p = counts / len(labels)
    return -np.sum(p*np.log2(p))

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    n = len(examples)
    labels = [example[attr] for example in examples]
    sum_entropy = 0
    for label in list(set(labels)):
        labels_tgt = [example[target_attr] for example in examples if example[attr] == label]
        H = calculate_entropy(labels_tgt)
        sum_entropy += len(labels_tgt) / n * H
    H_d = calculate_entropy([example[target_attr] for example in examples])
    return H_d - sum_entropy

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    labels = np.array([example[target_attr] for example in examples])
    unique_values, counts = np.unique(labels, return_counts=True)
    i_majority = np.argmax(counts)
    return unique_values[i_majority]


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    if len(set([example[target_attr] for example in examples])) == 1:
        return examples[0][target_attr]
    elif not attributes:
        return majority_class(examples, target_attr)

    # Select root
    ig_attributes = [calculate_information_gain(examples, attr, target_attr) for attr in attributes]
    i_highest_ig = np.argmax(np.array(ig_attributes))
    root = attributes[i_highest_ig]

    # Split dataset
    values_root = sorted(list(set([example[root] for example in examples])))
    tree = {}
    tree[root] = {}
    for value in values_root: # create the branches
        subset = [example for example in examples if example[root] == value]
        attributes_remaining = attributes.copy()
        attributes_remaining.remove(root)
        tree[root][value] = learn_decision_tree(subset, attributes_remaining, target_attr)

    return tree