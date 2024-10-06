def standard_scaler_fit(X):
    mean = []
    std = []
    for i in range(len(X[0])):
        col = [row[i] for row in X]
        mean_i = sum(col) / len(col)
        std_i = (sum((x - mean_i) ** 2 for x in col) / len(col)) ** 0.5
        mean.append(mean_i)
        std.append(std_i)
    return mean, std

def standard_scaler_transform(X, mean, std):
    X_scaled = []
    for row in X:
        scaled_row = [(x - m) / s if s != 0 else 0 for x, m, s in zip(row, mean, std)]
        X_scaled.append(scaled_row)
    return X_scaled

def min_max_scaler_fit(X):
    if not X:
        return [], []
    X_T = list(zip(*X))
    min_vals = [min(col) for col in X_T]
    max_vals = [max(col) for col in X_T]
    return min_vals, max_vals

def min_max_scaler_transform(X, min_vals, max_vals, feature_range=(0, 1)):
    range_min, range_max = feature_range
    result = []
    for row in X:
        scaled_row = []
        for x, min_val, max_val in zip(row, min_vals, max_vals):
            if max_val == min_val:
                scaled_row.append(range_min)
            else:
                scaled_value = ((x - min_val) / (max_val - min_val)) * (range_max - range_min) + range_min
                scaled_row.append(scaled_value)
        result.append(scaled_row)
    return result

def median(lst):
    n = len(lst)
    sorted_lst = sorted(lst)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2
    else:
        return sorted_lst[mid]

def percentile(lst, percentile_rank):
    n = len(lst)
    sorted_lst = sorted(lst)
    k = (n - 1) * percentile_rank
    f = int(k)
    c = k - f
    if f + 1 < n:
        return sorted_lst[f] + c * (sorted_lst[f + 1] - sorted_lst[f])
    else:
        return sorted_lst[f]

def robust_scaler_fit(X):
    if not X:
        return [], []
    X_T = list(zip(*X))
    medians = []
    iqrs = []
    for col in X_T:
        col_list = list(col)
        med = median(col_list)
        q1 = percentile(col_list, 0.25)
        q3 = percentile(col_list, 0.75)
        iqr = q3 - q1
        medians.append(med)
        iqrs.append(iqr)
    return medians, iqrs

def robust_scaler_transform(X, medians, iqrs):
    result = []
    for row in X:
        scaled_row = [
            (x - med) / iqr if iqr != 0 else 0
            for x, med, iqr in zip(row, medians, iqrs)
        ]
        result.append(scaled_row)
    return result
