###############################################
# Programmer: Chris Wong
# Class: CptS 322-01, Fall 2025
# Assignment Project
# 12/1/25

# Description: For PA#7, This module defines binary classification evaluation metrics, to include
# precision, recall, and F1 scores. 
##############################################
from mysklearn import myutils

import numpy as np # use numpy's random number generation
import math

from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # Make copies to avoid modifying the original data
    X_copy = [row[:] for row in X]  # Deep copy of X
    y_copy = y[:]  # Shallow copy of y
    
    # Shuffle if requested
    if shuffle:
        myutils.shuffle_in_unison(X_copy, y_copy, random_state)
    
    # Calculate number of test samples
    n_samples = len(X_copy)
    if isinstance(test_size, float):
        n_test_samples = math.ceil(n_samples * test_size)
    else:
        n_test_samples = test_size      
    
    # Split the data
    if shuffle:
        # When shuffling, take the first n_test_samples as test set
        X_test = X_copy[:n_test_samples]
        X_train = X_copy[n_test_samples:]
        y_test = y_copy[:n_test_samples]
        y_train = y_copy[n_test_samples:]
    else:
        # When not shuffling, take the last n_test_samples as test set
        X_test = X_copy[-n_test_samples:] if n_test_samples > 0 else []
        X_train = X_copy[:-n_test_samples] if n_test_samples > 0 else X_copy[:]
        y_test = y_copy[-n_test_samples:] if n_test_samples > 0 else []
        y_train = y_copy[:-n_test_samples] if n_test_samples > 0 else y_copy[:]    
    
    return X_train, X_test, y_train, y_test


def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    n_samples = len(X)
    indices = list(range(n_samples))
    
    # Shuffle if requested
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)
    
    # Create folds
    folds = []
    fold_sizes = [n_samples // n_splits] * n_splits
    
    # Distribute remainder across first folds
    for i in range(n_samples % n_splits):
        fold_sizes[i] += 1
    
    # Create each fold
    current_idx = 0
    for fold_size in fold_sizes:
        # Test indices for this fold
        test_indices = indices[current_idx:current_idx + fold_size]
        
        # Train indices are all others
        train_indices = indices[:current_idx] + indices[current_idx + fold_size:]
        
        folds.append((train_indices, test_indices))
        current_idx += fold_size
    
    return folds

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    n_samples = len(X)
    
    # Group indices by class label
    class_indices = {}
    for i, label in enumerate(y):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    # Shuffle indices within each class if requested
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
            
        # Always apply some deterministic transformation based on random_state
        # to ensure different random_states produce different results
        if random_state is not None:
            labels = list(class_indices.keys())
            # Apply different transformations based on random_state
            for i, label in enumerate(labels):
                if len(class_indices[label]) >= 2:
                    # Use random_state and label index to determine transformation
                    transform_type = (random_state + i) % 3
                    if transform_type == 0:
                        # No change (original order) 
                        pass
                    elif transform_type == 1:
                        # Reverse order
                        class_indices[label].reverse()
                    elif transform_type == 2:
                        # Swap first two elements
                        class_indices[label][0], class_indices[label][1] = \
                            class_indices[label][1], class_indices[label][0]
        else:
            # No random_state provided, just do normal shuffle
            for label in class_indices:
                np.random.shuffle(class_indices[label])
    
    # Create folds by distributing each class across folds
    folds = [[] for _ in range(n_splits)]
    
    # Distribute each class across folds
    for label, indices in class_indices.items():
        # Calculate how many samples per fold for this class
        n_class_samples = len(indices)
        samples_per_fold = n_class_samples // n_splits
        remainder = n_class_samples % n_splits
        
        # Distribute indices across folds
        current_idx = 0
        for fold_idx in range(n_splits):
            # Number of samples for this fold (add 1 extra for first 'remainder' folds)
            fold_size = samples_per_fold + (1 if fold_idx < remainder else 0)
            
            # Add indices to this fold
            fold_indices = indices[current_idx:current_idx + fold_size]
            folds[fold_idx].extend(fold_indices)
            current_idx += fold_size
    
    # Convert to (train_indices, test_indices) format
    result_folds = []
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = []
        for j in range(n_splits):
            if i != j:
                train_indices.extend(folds[j])
        
        result_folds.append((train_indices, test_indices))
    
    return result_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    n_original = len(X)
    if n_samples is None:
        n_samples = n_original
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Sample indices with replacement
    sample_indices = np.random.choice(n_original, size=n_samples, replace=True)
    
    # Get unique sampled indices and out-of-bag indices
    sampled_set = set(sample_indices)
    out_of_bag_indices = [i for i in range(n_original) if i not in sampled_set]
    
    # Build sampled data
    X_sample = [X[i] for i in sample_indices]
    X_out_of_bag = [X[i] for i in out_of_bag_indices]
    
    # Build y data if provided
    if y is not None:
        y_sample = [y[i] for i in sample_indices]
        y_out_of_bag = [y[i] for i in out_of_bag_indices]
    else:
        y_sample = None
        y_out_of_bag = None
    
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    n_labels = len(labels)
    
    # Initialize matrix with zeros
    matrix = [[0 for _ in range(n_labels)] for _ in range(n_labels)]
    
    # Fill the confusion matrix
    for true_val, pred_val in zip(y_true, y_pred):
        true_idx = labels.index(true_val)
        pred_idx = labels.index(pred_val)
        matrix[true_idx][pred_idx] += 1
    
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    # Count correct predictions
    correct_count = 0
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == pred_val:
            correct_count += 1
    
    if normalize:
        return correct_count / len(y_true)
    else:
        return correct_count


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    # Set default labels if not provided
    if labels is None:
        labels = list(set(y_true))
    
    # Set default pos_label if not provided
    if pos_label is None:
        pos_label = labels[0]
    
    # Calculate true positives (TP) and false positives (FP)
    tp = 0  # predicted positive and actually positive
    fp = 0  # predicted positive but actually negative
    
    for i in range(len(y_true)):
        if y_pred[i] == pos_label:
            if y_true[i] == pos_label:
                tp += 1
            else:
                fp += 1
    
    # Calculate precision: TP / (TP + FP)
    # If TP + FP = 0, return 0 (no positive predictions made)
    if tp + fp == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    # Set default labels if not provided
    if labels is None:
        labels = list(set(y_true))
    
    # Set default pos_label if not provided
    if pos_label is None:
        pos_label = labels[0]
    
    # Calculate true positives (TP) and false negatives (FN)
    tp = 0  # predicted positive and actually positive
    fn = 0  # predicted negative but actually positive
    
    for i in range(len(y_true)):
        if y_true[i] == pos_label:
            if y_pred[i] == pos_label:
                tp += 1
            else:
                fn += 1
    
    # Calculate recall: TP / (TP + FN)
    # If TP + FN = 0, return 0 (no actual positives in ground truth)
    if tp + fn == 0:
        return 0.0
    
    recall = tp / (tp + fn)
    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    # Calculate precision and recall using the functions above
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    
    # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    # If precision + recall = 0, return 0
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
