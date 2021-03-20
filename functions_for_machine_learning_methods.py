import numpy as np
from PIL import Image
import sys
import util
import cv2
import random
from sklearn.decomposition import PCA
import pandas as pd
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
train_data_dir = 'Train/Train/'
train_gt_dir = 'metadataTrain.csv'
test_data_dir = 'Test/Test/'

random.seed(123456)

# read training data
def read_train_data(img_size = 256, label_name = "ABNORMAL") :
    print("read training data")
    imgs = []
    gts = []
    
    gt_data = pd.read_csv(train_gt_dir)

    for idx, img_name in enumerate(gt_data["ID"]):
        img_path = train_data_dir + str(img_name) + '.bmp'
        seg1_path = train_data_dir + str(img_name) + '_segCyt.bmp'
        seg2_path = train_data_dir + str(img_name) + '_segNuc.bmp'
        
        img = cv2.imread(img_path)
        img_seg1 = cv2.imread(seg1_path)
        img_seg2 = cv2.imread(seg2_path)
        img = img*((img_seg1>0) + (img_seg2>0))
        img = cv2.resize(img, (img_size, img_size))
        img = img.reshape(-1)
        imgs.append(img)

        gts.append(gt_data[label_name][idx])  

    imgs = np.array(imgs)
    gts = np.array(gts)
    gts = gts.reshape(-1,1)
    print("training data shape: " + str(imgs.shape))
    return imgs, gts     

# read test data
def read_test_data(img_size = 256) :
    print("\nread test data")
    imgs = []

    img_names = util.io.ls(test_data_dir, '.bmp')

    imgs = []
    imgs_path = []
    for idx, img_name in enumerate(img_names):
        name, _ = os.path.splitext(img_name)
        if (name.isdigit()) :
            img_path = test_data_dir + name+'.bmp'
            seg1_path = test_data_dir + name + '_segCyt.bmp'
            seg2_path = test_data_dir + name + '_segNuc.bmp'
            imgs_path.append(img_path)
            
            img = cv2.imread(img_path)
            img_seg1 = cv2.imread(seg1_path)
            img_seg2 = cv2.imread(seg2_path)
            img = img*((img_seg1>0) + (img_seg2>0))
            img = cv2.resize(img, (img_size, img_size))
            img = img.reshape(-1)
            imgs.append(img)

    imgs = np.array(imgs)
    print("test data shape: " + str(imgs.shape))
    return imgs, imgs_path

# use PCA to select important features
def pre_process(X_train, X_test, y_train, num_component = 500) :
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
    pca = PCA(n_components=num_component,svd_solver='randomized', whiten=True)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print("After pre processing, training data shape: " + str(X_train_pca.shape))
    print("After pre processing, test data shape: " + str(X_test_pca.shape))
    return X_train_pca, X_test_pca, y_train

def write_data(y_pre, label_name, data_path, save_path):
    # write results
    print("write results")
    util.io.write_lines(save_path, "ID"+","+label_name+'\n', 'w')
    for idx, result in enumerate(y_pre):
        image_name = data_path[idx].split('/')[-1].split('.')[0]
        util.io.write_lines(save_path, image_name+","+str(result)+'\n', 'a')