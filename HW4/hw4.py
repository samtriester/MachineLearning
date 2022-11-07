#!/usr/bin/python3
# Homework 4 Code
import random

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

def oob(X_train, y_train, tree, oob_array, point_array):
    # TODO oob array is now updated vote count
    # use sign of oob array for error
    # iterate over point_array
    div=np.count_nonzero(oob_array)
    if div==0:
        div=np.count_nonzero(point_array)
    for i in range(0, point_array.size):

        if(point_array[i]==1):
            oob_array[i]=oob_array[i]+int(tree.predict(np.transpose(X_train[i].reshape(X_train.shape[1],1))))
    test_guess=np.sign(oob_array)
    return ((test_guess.size-np.count_nonzero(test_guess+y_train))/div)
def single_decision_tree(X_train, y_train, X_test, y_test):
    clf=DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train,y_train)
    train_guess=clf.predict(X_train)
    test_guess=clf.predict(X_test)
    train_error=((train_guess.size-np.count_nonzero(train_guess+y_train))/train_guess.size)
    test_error=((test_guess.size-np.count_nonzero(test_guess+y_test))/test_guess.size)
    return train_error, test_error
def bagged_trees(X_train, y_train, X_test, y_test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees 
    # and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: You may use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function
    oob_array=[0]*y_train.size
    out_of_bag_error =0
    clf = DecisionTreeClassifier(criterion='entropy')
    oob_plot=[]
    X=np.zeros_like(X_train)
    y=np.zeros_like(y_train)
    error_array = np.zeros_like(y_test)
    for i in range(0,num_bags):
        ones=np.ones_like(y_train)
        for j in range(0,X_train.shape[0]):
            rand=int(random.uniform(0, X_train.shape[0]))
            X[j]=X_train[rand]
            y[j]=y_train[rand]
            ones[rand]=0

        clf.fit(X, y)
        error_array = error_array + (clf.predict(X_test))
        out_of_bag_error =oob(X_train,y_train,clf,oob_array, ones)
        oob_plot.append(out_of_bag_error)
    test_guess=np.sign(error_array)
    test_error=((test_guess.size-np.count_nonzero(test_guess+y_test))/test_guess.size)
    plt.plot(list(range(1,201)), oob_plot)
    plt.title('Number of Bags vs. Out of Bag Error')
    plt.xlabel('Number of Bags')
    plt.ylabel('Out of Bag Error')
    plt.show()
    return out_of_bag_error, test_error

def main_hw4():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    num_bags = 200

    # Split data
    test1s=og_test_data[og_test_data[:, 0] == 1, :]
    test3s=og_test_data[og_test_data[:, 0] == 3, :]
    test5s=og_test_data[og_test_data[:, 0] == 5, :]
    testoneand3=np.append(test1s,test3s,axis=0)
    test3and5=np.append(test5s,test3s,axis=0)
    only1s=og_train_data[og_train_data[:, 0] == 1, :]
    only3s=og_train_data[og_train_data[:, 0] == 3, :]
    only5s=og_train_data[og_train_data[:, 0] == 5, :]
    oneand3=np.append(only1s,only3s,axis=0)
    threeand5=np.append(only3s,only5s, axis=0)
    X_train1and3 =np.append(np.ones((oneand3.shape[0],1)),oneand3[:,1:],axis=1)
    y_train1and3 =np.where(oneand3[:, 0]==3,-1,oneand3[:, 0])
    X_test1and3 =np.append(np.ones((testoneand3.shape[0],1)),testoneand3[:,1:],axis=1)
    y_test1and3 =np.where(testoneand3[:, 0]==3,-1,testoneand3[:, 0])
    X_train5and3 = np.append(np.ones((threeand5.shape[0],1)),threeand5[:,1:],axis=1)
    y_train5and3 = np.where(threeand5[:, 0]==3,-1,1)
    X_test5and3 = np.append(np.ones((test3and5.shape[0],1)),test3and5[:,1:],axis=1)
    y_test5and3 = np.where(test3and5[:, 0]==3,-1,1)

    # Run bagged trees
    out_of_bag_error, test_error = bagged_trees(X_train1and3, y_train1and3, X_test1and3, y_test1and3, num_bags)
    train_error, test_error_single = single_decision_tree(X_train1and3, y_train1and3, X_test1and3, y_test1and3)
    print("One and Three OOB: "+str(out_of_bag_error)+"\nTest Error: "+ str(test_error)+"\nSingle Train Error: "+str(train_error)+"\nTest Error: "+str(test_error_single))
    out_of_bag_error, test_error = bagged_trees(X_train5and3, y_train5and3, X_test5and3, y_test5and3, num_bags)
    train_error, test_error_single = single_decision_tree(X_train5and3, y_train5and3, X_test5and3, y_test5and3)
    print("Three and Five OOB: " + str(out_of_bag_error) + "\nTest Error: " + str(test_error) + "\nSingle Train Error: " + str(train_error) + "\nTest Error: " + str(test_error_single))


if __name__ == "__main__":
    error_array=0
    main_hw4()

