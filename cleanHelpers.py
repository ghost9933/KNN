



def replace_nan_and_question(X, y=None):
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j] == '?' or X[i][j] == '' or (type(X[i][j])==str  and X[i][j].lower() == 'nan'):
                X[i][j] = None
        if y is not None:
            if y[i] == '?' or y[i] == '' or y[i].lower() == 'nan':
                y[i] = None
    if y:
        return X, y
    else:
        return X

def replace_none_with_most_frequent(X):
    X_transposed = list(zip(*X))
    for i, feature in enumerate(X_transposed):
        non_none_values = [int(x) for x in feature if x is not None]
        if non_none_values:
            most_frequent_value = max(set(non_none_values), key=non_none_values.count)
            X_transposed[i] = [most_frequent_value if x is None else int(x) for x in feature]
        else:
            X_transposed[i] = [0 if x is None else int(x) for x in feature]
    return [list(row) for row in zip(*X_transposed)]

def replace_none_with_mode(y):
    non_none_values = [float(value) for value in y if value is not None]
    if non_none_values:
        mode_value = max(set(non_none_values), key=non_none_values.count)
        y = [mode_value if value is None else float(value) for value in y]
    else:
        y = [0.0 if value is None else float(value) for value in y]
    return y

def map_metric_to_sklearn(metric):
    metric_mapping = {
        'euclidean': 'euclidean',
        'manhattan': 'manhattan',
        'chebyshev': 'chebyshev',
        'cosine': 'cosine',
        'minkowski_3': ('minkowski', 3),
        'minkowski_4': ('minkowski', 4),
        'minkowski_5': ('minkowski', 5),
        'minkowski_10': ('minkowski', 10)
    }

    return metric_mapping.get(metric, 'euclidean')


import random

def dataSplit(X, y, test_size=0.25, stratify=None, random_state=None):
    """
    Splits the dataset into training and testing subsets.

    Parameters:
    - X: list of feature vectors
    - y: list of labels corresponding to X
    - test_size: float between 0.0 and 1.0, proportion of the dataset to include in the test split
    - stratify: array-like, default=None
        If not None, data is split in a stratified fashion, using this as class labels.
    - random_state: int, default=None
        Controls the shuffling applied to the data before applying the split.

    Returns:
    - X_train: list of training feature vectors
    - X_test: list of testing feature vectors
    - y_train: list of training labels
    - y_test: list of testing labels
    """
    if random_state is not None:
        random.seed(random_state)

    if len(X) != len(y):
        raise ValueError("The length of X and y must be equal.")

    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    if stratify is None:
        # Shuffle the indices
        indices = list(range(n_samples))
        random.shuffle(indices)

        # Split indices
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    else:
        # Stratified split
        # Create a dictionary mapping each class to the indices of samples with that class
        label_indices = {}
        for idx, label in enumerate(stratify):
            # Handle multi-label cases where label can be a list or tuple
            if isinstance(label, (list, tuple)):
                label = tuple(sorted(label))  # Convert to a hashable type
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(idx)

        train_indices = []
        test_indices = []

        # Split indices for each class
        for label, indices in label_indices.items():
            n_samples_label = len(indices)
            n_test_label = max(1, int(n_samples_label * test_size))
            n_train_label = n_samples_label - n_test_label

            # Shuffle indices for this class
            random.shuffle(indices)

            # Split indices for this class
            test_indices.extend(indices[:n_test_label])
            train_indices.extend(indices[n_test_label:])

    # Optionally shuffle the train and test indices
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    # Create train and test splits
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test
