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