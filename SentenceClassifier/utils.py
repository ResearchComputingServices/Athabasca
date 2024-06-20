import pickle
import os

import numpy as np

from umap import UMAP

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing as p

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_umap_transformer(umap_transformer : UMAP,
                          file_path : str) -> None:
    try:
        pickle.dump(umap_transformer, open(file_path, 'wb'))
    except pickle.PickleError as e:
        raise
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_umap_transformer(file_path : str) -> UMAP:
    return pickle.load((open(file_path, 'rb')))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_lr_classifier( lr_classifier : LogisticRegression,
                        file_path : str) -> None:
    try:
        pickle.dump(lr_classifier, open(file_path, 'wb'))
    except pickle.PickleError as e:
        raise

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_lr_classifier(file_path : str) -> LogisticRegression:
    return  pickle.load((open(file_path, 'rb')))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_min_max_scaler(min_max_scaler : p.MinMaxScaler,
                        file_path : str) -> None:
    try:
        pickle.dump(min_max_scaler, open(file_path, 'wb'))
    except pickle.PickleError as e:
        raise
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_min_max_scaler(file_path : str) -> p.MinMaxScaler:
    return  pickle.load((open(file_path, 'rb'))) 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def if_not_exist_create_dir(dir_path : str) -> None:
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        # directory already exists
        pass

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~