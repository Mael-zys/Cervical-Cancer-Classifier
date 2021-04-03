import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from PIL import Image
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.feature_selection import (RFE, RFECV, SelectKBest, chi2,
                                       f_classif, f_regression)
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.utils import shuffle
import shap
import util


# feature selection
# pca_mode can be {None, "pca", "kpca", "spca"}
def select_features(X_train, X_test, y_train, select_feature = None, n_component = 500) :
    
    if select_feature is not None:
        print("feature selection mode is: " + select_feature)
    
    # pca
    if select_feature == "pca":
        pca = PCA(n_components=n_component,svd_solver='randomized', whiten=True)
        pca.fit(X_train)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    # Kernel Pca
    elif select_feature == "kpca":
        kpca = KernelPCA(n_components=n_component, kernel='rbf', gamma=2, n_jobs=8)
        kpca.fit(X_train)

        X_train = kpca.transform(X_train)
        X_test = kpca.transform(X_test)

    # Sparse Pca
    elif select_feature == "spca":
        spca = SparsePCA(n_components=n_component, n_jobs=8)
        spca.fit(X_train)

        X_train = spca.transform(X_train)
        X_test = spca.transform(X_test)

    # Variable Ranking
    elif select_feature == "select_best":
        bestfeatures = SelectKBest(score_func=f_classif, k=n_component)
        bestfeatures.fit(X_train, y_train)

        X_train = bestfeatures.transform(X_train)
        X_test = bestfeatures.transform(X_test)

    # Built-in Feature Importance: RF
    elif select_feature == "RF":
        model = RandomForestClassifier(n_jobs=8, random_state=0)
        model.fit(X_train, y_train)
        model.feature_importances_

    # Built-in Feature Importance: ExtraTreesClassifier
    elif select_feature == "ExtraTrees":
        model = ExtraTreesClassifier(n_jobs=8, random_state=0)
        model.fit(X_train, y_train)
        model.feature_importances_

    # shap
    elif select_feature == "shap":
        model = RandomForestClassifier(n_jobs=8, random_state=0)
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values_train = explainer.shap_values(X_train)
        shap_values_test = explainer.shap_values(X_test)

    # Recursive Feature Elimination
    elif select_feature == "RFE":
        # Instantiate RFECV visualizer with a random forest regressor
        rfecv = RFECV(RandomForestRegressor())

        rfecv.fit(X_train, y_train) # Fit the data to the visualizer

        print("Optimal number of features : %d" % rfecv.n_features_)

    # Sequential Feature Selection
    elif select_feature == "SFS":
        # Build RF regressor to use in feature selection
        clf = RandomForestRegressor()

        # Sequential Forward Selection
        sfs = sfs(clf,
                k_features=5, 
                forward=True,
                floating=False,
                verbose=2,
                scoring='neg_mean_squared_error',
                cv=5)

        sfs = sfs.fit(X_train, y_train)

        print('\nSequential Forward Selection (k=5):')
        print(sfs.k_feature_idx_)
        print('CV Score:')
        print(sfs.k_score_)

    # Permutation importance
    elif select_feature == "permutation":
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        result = permutation_importance(rf, X, y, n_repeats=10,
                                        random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()

    return X_train, X_test
