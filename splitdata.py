import random
from collections import defaultdict

def stratified_train_test_split(X, y, test_size=0.25, random_state=None):
    print("spliting the data with maintaining the class balance",test_size,random_state)
    if random_state is not None:
        random.seed(random_state)
    data_by_class = defaultdict(list)
    for features, label in zip(X, y):
        data_by_class[label].append(features)


    X_train, X_test, y_train, y_test = [], [], [], []
    for label, features in data_by_class.items():
        random.shuffle(features)
        split_index = int(len(features) * (1 - test_size))
        X_train_class = features[:split_index]
        X_test_class = features[split_index:]
        y_train_class = [label] * len(X_train_class)
        y_test_class = [label] * len(X_test_class)
        X_train.extend(X_train_class)
        X_test.extend(X_test_class)
        y_train.extend(y_train_class)
        y_test.extend(y_test_class)
    combined_train = list(zip(X_train, y_train))
    combined_test = list(zip(X_test, y_test))

    random.shuffle(combined_train)
    random.shuffle(combined_test)

    X_train, y_train = zip(*combined_train)
    X_test, y_test = zip(*combined_test)

    return list(X_train), list(X_test), list(y_train), list(y_test)

