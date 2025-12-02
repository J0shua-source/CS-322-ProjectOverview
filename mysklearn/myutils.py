import numpy as np 

def shuffle_in_unison(X, y, random_state=None):
    """Shuffle X and y in unison to maintain parallel order.
    
    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
        random_state(int): integer used for seeding a random number generator for reproducible results
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Create indices and shuffle them
    indices = list(range(len(X)))
    np.random.shuffle(indices)
    
    # Reorder X and y based on shuffled indices
    X[:] = [X[i] for i in indices]
    y[:] = [y[i] for i in indices]

def print_confusion_matrix_formatted(cm, labels, subset_name):
    """Print a nicely formatted confusion matrix.
    
    Args:
        cm(list of list of int): The confusion matrix
        labels(list of str): The class labels
        subset_name(str): Name/description of the confusion matrix to display
    """
    print(f"\n{subset_name}")
    print("-" * 50)
    
    # Header
    header = "Actual/Predicted |"
    for label in labels:
        header += f" {label:>10s} |"
    header += " Total"
    print(header)
    print("-" * len(header))
    
    # Rows
    for i, label in enumerate(labels):
        row = f"{label:>16s} |"
        row_sum = sum(cm[i])
        for val in cm[i]:
            row += f" {val:>10d} |"
        row += f" {row_sum:>5d}"
        print(row)
    
    # Column totals
    col_totals = "Total            |"
    grand_total = 0
    for j in range(len(labels)):
        col_sum = sum(cm[i][j] for i in range(len(labels)))
        col_totals += f" {col_sum:>10d} |"
        grand_total += col_sum
    col_totals += f" {grand_total:>5d}"
    print("-" * len(header))
    print(col_totals)