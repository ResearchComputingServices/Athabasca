import pickle

import matplotlib.pyplot as plt

import numpy as np

from umap import UMAP

from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay

from DataSet import DataSet

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_umap_transformer(embeddings : np.ndarray,
                            targets = None,
                            n_components=2,
                            metric = 'cosine',
                            min_dist = 0.) -> UMAP:
    
    umap_transformer = UMAP(n_components=n_components,
                            metric=metric,
                            min_dist=min_dist)
    
    umap_transformer.fit(X=embeddings, 
                         y=targets)
    
    return umap_transformer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_umap_transformer(umap_transformer : UMAP,
                          file_path : str) -> None:
    
    pickle.dump(umap_transformer, open(file_path, 'wb'))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_umap_transformer(file_path : str) -> UMAP:
    return  pickle.load((open(file_path, 'rb')))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_lr_classifier( lr_classifier : LogisticRegression,
                        file_path : str) -> None:
    
    pickle.dump(lr_classifier, open(file_path, 'wb'))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_lr_classifier(file_path : str) -> LogisticRegression:
    return  pickle.load((open(file_path, 'rb')))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_classifier(lr_classifier : LogisticRegression, 
                    data_set : DataSet,
                    verbose = False):
        
    total = len(data_set.data_list)
    
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
       
    for i, test_sample in enumerate(data_set.data_list):
        formatted_sample = np.array(test_sample.reduced_encoding).reshape(1, -1)
    
        probs = lr_classifier.predict_proba(formatted_sample)
    
        predicted_class_index = lr_classifier.predict(formatted_sample)
        predicted_class_label = data_set.get_label_from_index(predicted_class_index)
            
        actual_class_label = test_sample.label_string
        actual_class_index = test_sample.label_index
                
        if actual_class_index == predicted_class_index:
            if actual_class_index == 1:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if actual_class_index > predicted_class_index:
                false_neg += 1
            else:
                false_pos += 1

        if verbose and predicted_class_index != actual_class_index:
            print(f'{i} of {total}')
            print(test_sample.sentence)
            print(f'Actual Class: {actual_class_label} {actual_class_index}')
            print(f'Predicted: {predicted_class_label} {predicted_class_index}')
            print(probs)
            input()      
    
        # if predicted_class_index != actual_class_index and actual_class_index == 0:
        #     print(test_sample.sentence)
    
    num_neg = len(data_set.get_data_with_label_index(0))
    num_pos = len(data_set.get_data_with_label_index(1))
    
    prec = true_pos / (true_pos + false_pos)
    accu = (true_pos + true_neg)/ total
    reca = true_pos / (true_pos + false_neg)
        
    print(f'True +: {true_pos}')
    print(f'True -: {true_neg}')
    print(f'False +: {false_pos}')
    print(f'False -: {false_neg}')
    print(f'Precision: {prec}')    
    print(f'Accuracy: {accu}')
    print(f'Recall: {reca}')
    print(f'F1-Score: {(2*reca*prec/(reca+prec))}')
    print(f'total: {total}')
    print(f'# +: {num_pos}')
    print(f'# -: {num_neg}')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~