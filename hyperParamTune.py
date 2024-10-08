

# custom import
from myCustomKNN import CustomKNN
from sklearn.metrics import accuracy_score
from DBknn import dbKNN
from itertools import product




def custom_knn_grid_search(X_train, y_train, X_test, y_test, param_grid):
    best_params = None
    best_accuracy = 0
    best_model = None

    for n_neighbors in param_grid['n_neighbors']:
        for weights in param_grid['weights']:
            for metric in param_grid['metric']:
                knn_custom = CustomKNN(n=n_neighbors, task='classification', weights=weights, metric=metric)
                knn_custom.fit(X_train, y_train)
                y_pred_custom = knn_custom.predict(X_test)
                accuracy_custom = accuracy_score(y_test, y_pred_custom)

                if accuracy_custom > best_accuracy:
                    best_accuracy = accuracy_custom
                    best_params = {'n_neighbors': n_neighbors, 'weights': weights, 'metric': metric}
                    best_model = knn_custom

    return best_model, best_params, best_accuracy


def dbKNN_knn_grid_search(X_train, y_train, X_test, y_test, param_grid):
    best_params = None
    best_accuracy = 0
    best_model = None

    for n_neighbors in param_grid['n_neighbors']:
        for weights in param_grid['weights']:
            for metric in param_grid['metric']:
                knn_custom = dbKNN(n=n_neighbors, task='classification', weights=weights, metric=metric)
                knn_custom.fit(X_train, y_train)
                y_pred_custom = knn_custom.predict(X_test)
                accuracy_custom = accuracy_score(y_test, y_pred_custom)

                if accuracy_custom > best_accuracy:
                    best_accuracy = accuracy_custom
                    best_params = {'n_neighbors': n_neighbors, 'weights': weights, 'metric': metric}
                    best_model = knn_custom

    return best_model, best_params, best_accuracy

