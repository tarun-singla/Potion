import multiprocessing
import numpy as np
import csv

def import_data():
    X = np.genfromtxt("train_X_lg_v2.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_lg_v2.csv", delimiter=',', dtype=np.float64)
    return X, Y

def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s

def compute_cost(X, Y, W, b):
    m = len(X)
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    A[A == 1] = 0.99999
    A[A == 0] = 0.00001
    cost = -(1/m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A)))
    return cost

def compute_gradient_of_cost_function(X, Y, W, b):
    m = len(X)
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    dW = (1/m) * (np.dot((A - Y).T, X))
    db = 1/m * np.sum(A-Y)
    dW = dW.T
    return dW, db

def optimize_weights_using_gradient_descent(X, Y, W, b, learning_rate):
    previous_iter_cost = 0
    iter_no = 0
    while True:
        iter_no += 1
        dW, db = compute_gradient_of_cost_function(X, Y, W, b)
        W = W - (learning_rate * dW)
        b = b - (learning_rate * db)
        cost = compute_cost(X, Y, W, b)
        if abs(previous_iter_cost - cost) < 0.000001:
            print(iter_no, cost)
            break
        #if iter_no % 1000 == 0:
        #print(iter_no, cost)
        previous_iter_cost = cost
    return W, b

def get_train_data_for_class(train_X, train_Y, class_label):
    class_X = np.copy(train_X)
    class_Y = np.copy(train_Y)
    class_Y = np.where(class_Y == class_label, 1, 0)
    return class_X, class_Y

def train_model(X, Y):
    X, Y = get_train_data_for_class(X, Y, 3)
    # X = np.insert(X, 0, 1, axis=1)
    # Y = Y.reshape(len(X), 1)
    W = np.ones((X.shape[1], ))
    b = 1
    W, b = optimize_weights_using_gradient_descent(X, Y, W, b, 0.0075)
    W = np.append(W, b)
    return W

def save_model(weights, weights_file_name):
    with open(weights_file_name, 'a', newline='') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(map(lambda x: [x], weights))
        weights_file.close()

if __name__ == "__main__":
    X, Y = import_data()
    weights = train_model(X, Y)
    save_model(weights, "WEIGHTS_FILE.csv")
