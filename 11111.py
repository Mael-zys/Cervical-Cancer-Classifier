import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
import sys
import util
import cv2
import random
from sklearn.decomposition import PCA
import pandas as pd
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from functions_for_machine_learning_methods import read_train_data, read_test_data, pre_process, write_data

def SVM_train_pre(X_train, X_test, y_train) :
    print("\nSVM estimator")
    
    # gaussian kernel
    p_grid_lsvm = {'C': [1e-3,1e-2,1e-1,1,5,1e1,1e2],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1], }
    Lsvm = SVC(kernel='rbf')
    grid_lsvm = GridSearchCV(estimator=Lsvm, param_grid=p_grid_lsvm, scoring="f1", cv=5)
    grid_lsvm.fit(X_train, y_train)
    print("Best training Score: {}".format(grid_lsvm.best_score_))
    print("Best training params: {}".format(grid_lsvm.best_params_))
    y_pre = grid_lsvm.predict(X_test)
    return y_pre

def random_forest_train_pre(X_train, X_test, y_train):
    print("\nRandom forest estimator")
    
    # Random forest
    RF=RandomForestClassifier(random_state=0)
    p_grid_RF = {'n_estimators': [10,15,20,25,30], 'min_samples_leaf': [2,3,4,5,6], 'max_features': ['sqrt','log2']}   

    grid_RF = GridSearchCV(estimator=RF, param_grid=p_grid_RF, scoring='f1', cv=5)
    grid_RF.fit(X_train, y_train.ravel())

    print("Best Validation Score: {}".format(grid_RF.best_score_))
    print("Best params: {}".format(grid_RF.best_params_))
    y_pre = grid_RF.predict(X_test)
    return y_pre

def bagging_train_pre(X_train, X_test, y_train):
    print("\nBagging estimator")
    
    Tree = DecisionTreeClassifier(random_state=0)
    p_grid_tree = {'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'min_samples_leaf':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]} 
    grid_tree = GridSearchCV(Tree, p_grid_tree, cv=5)
    grid_tree.fit(X_train, y_train)
    best_params = grid_tree.best_params_
    
    Tree = DecisionTreeClassifier(min_samples_leaf=best_params["min_samples_leaf"],min_samples_split=best_params["min_samples_split"], random_state=0)
    p_grid_bagging = {'n_estimators': [5,10,15,20]}      
    bag=BaggingClassifier(base_estimator=Tree, random_state=0)
    grid_bagging = GridSearchCV(estimator=bag, param_grid=p_grid_bagging, cv=5, scoring='f1')
    grid_bagging.fit(X_train, y_train.ravel())
    print("Best Validation Score: {}".format(grid_bagging.best_score_))
    print("Best params: {}".format(grid_bagging.best_params_))
    y_pre = grid_bagging.predict(X_test)
    return y_pre

def logistic_train_pre(X_train, X_test, y_train) :
    print("\nLogistic estimator")
    
    # Logistic
    logi=LogisticRegression()
    p_grid_log = {'C': [1e-1,0.5,1,5,1e1]}  

    grid_log = GridSearchCV(estimator=logi, param_grid=p_grid_log, scoring='f1', cv=5)
    grid_log.fit(X_train, y_train.ravel())

    print("Best Validation Score: {}".format(grid_log.best_score_))
    print("Best params: {}".format(grid_log.best_params_))
    y_pre = grid_log.predict(X_test)
    return y_pre

def KNN_train_pre(X_train, X_test, y_train) :
    print("\nKNN estimator")
    
    # KNN
    KNN=KNeighborsClassifier()
    p_grid_knn = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,15]}

    grid_knn = GridSearchCV(estimator=KNN, param_grid=p_grid_knn, scoring='f1', cv=5)
    grid_knn.fit(X_train, y_train.ravel())

    print("Best Validation Score: {}".format(grid_knn.best_score_))
    print("Best params: {}".format(grid_knn.best_params_))
    y_pre = grid_knn.predict(X_test)
    return y_pre

if __name__ == '__main__': 
    # read training data 
    X_train, y_train = read_train_data(128)
    
    # read test data
    X_test, X_path = read_test_data(128)

    print("\npre processing")

    # Scale data (each feature will have average equal to 0 and unit variance)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)

    # shuffle
    indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[indices, :]
    y_train = y_train[indices, :]

    # PCA
    pca = PCA(n_components=200,svd_solver='randomized', whiten=True)
    pca.fit(X_train)
    print(np.sum(pca.explained_variance_ratio_))

