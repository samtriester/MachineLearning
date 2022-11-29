#!/usr/bin/python3
# Homework 5 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using decision stumps as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use
    weights = np.zeros(len(X_train))
    for i in range(len(X_train)):
        weights[i] = 1/len(X_train)

    classifiers = []
    alpha = []
    predictions = np.zeros([len(X_train), n_trees])

    for i in range(n_trees):
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
        clf.fit(X_train, y_train, sample_weight=weights.flatten())
        predictions[:, i] = clf.predict(X_train)
        error = 0
        for j in range(len(X_train)):
            if y_train[j] != predictions[j, i]:
                error += weights[j]
        a = 0.5*np.log((1-error)/error)
        classifiers.append(clf)
        alpha.append(a)
        gamma = np.sqrt((1-error)/error)
        z_t = gamma*error + (1/gamma)*(1-error)
        for j in range(len(X_train)):
            weights[j] = np.exp(-alpha[i]*predictions[j, i]*y_train[j])*weights[j]
        sum_d = np.sum(weights)
        for j in range(len(weights)):
            weights[j] = weights[j]/sum_d

    train_errors = 0
    test_errors = 0

    for i in range(len(X_train)):
        predictions = []
        for j in range(len(classifiers)):
            predictions.append(classifiers[j].predict(X_train[i].reshape(1, -1)) * alpha[j])
        sum = np.sum(predictions)
        classification = np.sign(sum)
        if classification != y_train[i]:
            train_errors = train_errors + 1
    train_errors = train_errors/len(X_train)
    for i in range(len(X_test)):
        predictions = []
        for j in range(len(classifiers)):
            predictions.append(classifiers[j].predict(X_test[i].reshape(1, -1)) * alpha[j])
        sum = np.sum(predictions)
        classification = np.sign(sum)
        if classification != y_test[i]:
            test_errors = test_errors + 1
    test_errors = test_errors/len(X_test)

    train_error = train_errors
    test_error = test_errors

    return train_error, test_error


def main_hw5():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    # Split data
    test1s = og_test_data[og_test_data[:, 0] == 1, :]
    test3s = og_test_data[og_test_data[:, 0] == 3, :]
    test5s = og_test_data[og_test_data[:, 0] == 5, :]
    testoneand3 = np.append(test1s, test3s, axis=0)
    test3and5 = np.append(test5s, test3s, axis=0)
    only1s = og_train_data[og_train_data[:, 0] == 1, :]
    only3s = og_train_data[og_train_data[:, 0] == 3, :]
    only5s = og_train_data[og_train_data[:, 0] == 5, :]
    oneand3 = np.append(only1s, only3s, axis=0)
    threeand5 = np.append(only3s, only5s, axis=0)
    X_train1and3 = np.append(np.ones((oneand3.shape[0], 1)), oneand3[:, 1:], axis=1)
    y_train1and3 = np.where(oneand3[:, 0] == 3, -1, oneand3[:, 0])
    X_test1and3 = np.append(np.ones((testoneand3.shape[0], 1)), testoneand3[:, 1:], axis=1)
    y_test1and3 = np.where(testoneand3[:, 0] == 3, -1, testoneand3[:, 0])
    X_train5and3 = np.append(np.ones((threeand5.shape[0], 1)), threeand5[:, 1:], axis=1)
    y_train5and3 = np.where(threeand5[:, 0] == 3, -1, 1)
    X_test5and3 = np.append(np.ones((test3and5.shape[0], 1)), test3and5[:, 1:], axis=1)
    y_test5and3 = np.where(test3and5[:, 0] == 3, -1, 1)
    num_trees=200

    # Run bagged trees
    # out_of_bag_error1, test_error_bag1 = bagged_trees(X_train1, y_train1, X_test1, y_test1, num_bags)
    # print(out_of_bag_error1, test_error_bag1)
    # train_error1, test_error1 = single_decision_tree(X_train1, y_train1, X_test1, y_test1)
    # print(train_error1, test_error1)



    total_train_errors = []
    total_test_errors = []
    for i in range(1, num_trees + 1):
        train_error, test_error = adaboost_trees(X_train5and3, y_train5and3, X_test5and3, y_test5and3, i)
        total_train_errors.append(train_error)
        total_test_errors.append(test_error)


    plt.plot(np.arange(1, num_trees + 1), total_train_errors, label='Training Error')
    plt.plot(np.arange(1, num_trees + 1), total_test_errors, label='Testing Error')
    plt.title("# of Trees compared to Errors, 3v5")
    plt.xlabel("#Trees")
    plt.ylabel("Error")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main_hw5()