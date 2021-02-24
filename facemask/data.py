import pandas as pd
from google.cloud import storage
import cv2
import glob
import numpy as np
from tensorflow.keras.utils import to_categorical


# BUCKET_NAME = ''
# BUCKET_TRAIN_DATA_PATH1 = 'Facemask_dataset/mask'
# BUCKET_TRAIN_DATA_PATH2 = 'Facemask_dataset/no_mask'
# BUCKET_TRAIN_DATA_PATH3 = 'Facemask_dataset/bad_mask'



def get_data():
    # client = storage.Client()
    local_path_mask = '/mnt/c/Users/user/Desktop/Facemask/Data/mask'
    local_path_no_mask = '/mnt/c/Users/user/Desktop/Facemask/Data/no_mask'
    local_path_bad_mask = '/mnt/c/Users/user/Desktop/Facemask/Data/bad_mask'
    list_files_mask = glob.glob(local_path_mask + '/*.jpg')
    list_files_no_mask = glob.glob(local_path_no_mask + '/*.png')
    list_files_bad_mask = glob.glob(local_path_bad_mask + '/*.jpg')
    X_mask_resized = []
    X_no_mask_resized = []
    X_bad_mask_resized = []

    # for i in range(len(list_files_mask)):
    for i in range(100):
        image = cv2.imread(list_files_mask[i])
        new_image = cv2.resize(image, (256, 256))
        X_mask_resized.append(np.array(new_image))

    # for i in range(len(list_files_no_mask)):
    for i in range(100):
        image = cv2.imread(list_files_no_mask[i])
        new_image = cv2.resize(image, (256, 256))
        X_no_mask_resized.append(np.array(new_image))

    # for i in range(len(list_files_bad_mask)):
    for i in range(100):
        image = cv2.imread(list_files_bad_mask[i])
        new_image = cv2.resize(image, (256, 256))
        X_bad_mask_resized.append(np.array(new_image))

    X_mask_resized = np.array(X_mask_resized)
    X_no_mask_resized = np.array(X_no_mask_resized)
    X_bad_mask_resized = np.array(X_bad_mask_resized)
    return X_mask_resized, X_no_mask_resized, X_bad_mask_resized

def create_dataframe(X1,X2,X3):
    nb_masks = X1.shape[0]
    nb_nomasks = X2.shape[0]
    nb_badmasks = X3.shape[0]

    X, y = [], []

    for i in range(nb_masks):
        X.append(X1[i])
        y.append(1)

    for i in range(nb_nomasks):
        X.append(X2[i])
        y.append(0)

    for i in range(nb_badmasks):
        X.append(X3[i])
        y.append(2)

    c = list(zip(X, y))
    np.random.shuffle(c)
    X, y = zip(*c)
    return np.array(X), np.array(y)

def dataset_preproc(X,y):
    X_norm=X/255
    y_cat=to_categorical(y)
    return X_norm,y_cat


if __name__ == '__main__':
    X1, X2, X3 = get_data()
    X, y = create_dataframe(X1,X2,X3)
