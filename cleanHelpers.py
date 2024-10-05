



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