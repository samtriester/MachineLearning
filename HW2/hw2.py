#!/usr/bin/python3
# Homework 2 Code
import numpy as np
import pandas as pd
import time







def find_cross_entropy_error(w, X, y):
    error_array = np.matmul(np.matmul(np.transpose(-y), np.transpose(w)), np.transpose(X))


    ones=np.ones_like(error_array)
    error_array = np.log(np.exp(error_array)+ones)

    return np.sum(error_array, axis=1) / y.size


def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # Your code here, assign the proper value to binary_error:
    # Get signs of prediction vector (C=50%)
    prediction = np.matmul(np.transpose(w), np.transpose(X))
    prediction = np.sign(prediction)
    # different predictions should cancel out
    total_error = prediction + y

    # count canceled outs and divide by total
    binary_error = (np.count_nonzero(total_error) / (y.size * y.size))
    return binary_error


def get_gradient(X, y, w):
    y_mult=np.repeat(y,y.size, axis=1)
    y=np.repeat(y,w.shape[0], axis=1)
    denom_array = np.matmul(np.matmul(-y, w), X)
    ones=np.ones_like(denom_array)
    denom_array = np.exp(denom_array)+ones
    num_array = np.matmul(y_mult, X)
    return -np.sum(np.divide(num_array, denom_array), axis=0) / y.size


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions; 
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:
    cont = True
    t = 0
    w = w_init
    change = 0
    while cont == True:
        t = t + 1
        change = get_gradient(X, y, np.repeat(w, y.size, axis=1)).reshape(13,1)

        w = w- eta * change
        if np.all(change < grad_threshold) or t > max_its:
            cont = False


    e_in = find_cross_entropy_error(np.repeat(w, y.size, axis=1), X, y)
    return t, w, e_in


def main():
    # Load training data
    train_data = pd.read_csv('clevelandtrain.csv')

    # Load test data
    test_data = pd.read_csv('clevelandtest.csv')
    train_X = train_data.loc[:, train_data.columns != 'heartdisease::category|0|1'].to_numpy()
    train_Y = train_data['heartdisease::category|0|1'].map({0: -1, 1: 1}).to_numpy().reshape(152, 1)
    w_init = np.zeros((13, 1))

    test_X = test_data.loc[:, test_data.columns != 'heartdisease::category|0|1'].to_numpy()
    test_Y = test_data['heartdisease::category|0|1'].map({0: -1, 1: 1}).to_numpy().reshape(145, 1)
    # Your code here
    tick=time.perf_counter()
    t, w, e_in = logistic_reg(train_X, train_Y, w_init, 10000, 0.00001 , .001 )
    tock=time.perf_counter()
    print("%8f seconds"%(tock-tick))
    print(w)
    print("In Sample Binary Error %6.5f" % (find_binary_error(w, train_X, train_Y)))
    print("Test Binary Error %6.5f" % (find_binary_error(w, test_X, test_Y)))
    print("Cross-Entropy Error %6.5f" % e_in)
    print(t)


if __name__ == "__main__":
    main()
