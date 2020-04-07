import numpy as np  
from sklearn.metrics import mean_squared_error

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def restore(normalized, original):
    return normalized * (np.max(original) - np.min(original)) + np.min(original)

def moving_average(y, n=3) :
    ret = np.cumsum(y, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def RMSE(true_vals, preds):
    return np.sqrt(mean_squared_error(true_vals, preds))

def train_test_split(data, ratio):
    train_size = int(ratio * len(data))

    train_data = data[:train_size]
    test_data = data[train_size:]
    
    return train_data, test_data

def cdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

# short tests
if __name__ == "__main__":
    x = np.arange(50)
    print(f'Original data:\n {x}\n')
    norm = normalize(x)
    print(f'Original data:\n {norm}\n')
    x = restore(norm, x)
    print(f'Original data:\n {x}\n')
    n = 3
    x_avg = moving_average(x, n)
    print(f'Moving average w/ n={n}:\n {x_avg}\n')
    trues = np.arange(0, 10, 1)
    preds = np.r_[np.arange(0, 9, 1), np.array([8])]
    print(trues)
    print(preds)
    print(f'RMSE: {RMSE(trues, preds)}\n')
    full = np.array([1, 2, 3, 4, 5])
    ratio = .8
    train, test = train_test_split(full, ratio)
    print(f'Data: {full}')
    print(f'Train data: {train}')
    print(f'Test data: {test}')
   