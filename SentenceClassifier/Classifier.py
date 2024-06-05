import _io
import os
import json
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from umap import UMAP
from sentence_transformers import SentenceTransformer

import numpy as np
import pandas as pd

import plotly_express as px
import matplotlib.pyplot as plt

from pprint import pprint

from .DataSet import DataSet
from .utils import if_not_exist_create_dir, save_lr_classifier, save_umap_transformer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# string literals used in the class

BASE_CLASSIFIER_JSON_FILE_NAME = 'base_classifier.json'
TRAINING_DATA_SET_JSON_FILE_NAME = 'training_data_set.json'
UMAP_PICKLE_FILE_NAME = 'umap.pkl'
LOG_REG_PICKLE_FILE_NAME = 'logreg.pkl'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SentenceClassifier:
    """Class which encapsulates all the return funtionality to perform
    train a sematic sentence classifier and use it for prediction
    """  
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self,
                 name = None,
                 pretrained_transformer_path = None,
                 verbose = True) -> None:

        self.name = name
        self.pretrained_transformer_path = pretrained_transformer_path

        self.logreg_classifier = LogisticRegression(verbose=verbose)

        self.training_data_path = None
        self.umap_transformer = None
        self.training_data_set = None
        self.is_initialized = False
        self.training_data_stream = None
        self.sentence_transformer = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def initialize(self)-> None:
        """Get the train set ready for use in training
        1. perform embedding
        2. train the umap transformer
        3. reduce the embeddings using umap
        """
 
        if self._check_training_data():    
            # load labelled training data from disk
            self.training_data_set = DataSet(file_stream=self.training_data_stream)
            
            if self._check_data_set():
                # use the pre-trained model to embedded the trianing data
                self._perform_embedding()

                # use the pre-trained embeddings to create the umap reduction transformer
                self._create_umap_transformer()

                # reduce the training data
                self._perform_reduction()

                self.is_initialized = True
            else:
                print('_check_data_set returned None')
                input()
        else:
            print('_check_training_data returned None')
            input()   
        # TODO: Add warning if no training data specified or no samples in data set        

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_train_data_path(self,
                            training_data_path : str) -> None:
        
        self.training_data_stream = open(training_data_path,'r',encoding='utf-8')
                   
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       
    def set_train_data_stream(  self,
                                training_data_stream : _io.BufferedReader) -> None:
        
        self.training_data_stream = training_data_stream

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def train_classifier(self) -> None:
        """trains the classifier on the data provided in data_set
        """

        if not self.is_initialized:
            self.initialize()

        if self._check_data_set():

            samples = self.training_data_set.get_reduced_embeddings()
            labels = self.training_data_set.get_label_index_list()

            self.logreg_classifier.fit( X=samples,
                                        y=labels)
            
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _create_umap_transformer(   self,
                                    n_components=2,
                                    metric = 'cosine',
                                    min_dist = 0.) -> None:

        embeddings = self.training_data_set.get_embeddings()
        targets = self.training_data_set.get_label_index_list()
 
        self.umap_transformer = UMAP(   n_components=n_components,
                                        metric=metric,
                                        min_dist=min_dist)

        self.umap_transformer.fit(  X=embeddings,
                                    y=targets)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _perform_embedding(self) -> None:
        if self._check_transformer_path():
            sent_trans = SentenceTransformer(self.pretrained_transformer_path)
            self.training_data_set.perform_embedding(sent_trans)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _perform_reduction(self) -> None:
        if self._check_umap():
            self.training_data_set.perform_reduction(self.umap_transformer)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
    def _check_training_data(self) -> bool:
        return self._check_training_data != None 

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _check_data_set(self) -> bool:
        return (self.training_data_set != None) and (self.training_data_set.n_samples > 0)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _check_umap(self) -> bool:
        return  self.umap_transformer != None
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    def _check_transformer_path(self) -> bool:
        return True

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def generate_interactive_plot(self) -> None:
                
        df = pd.DataFrame()
        df.insert(0, "Reduced Feature 1", self.training_data_set.get_reduced_embeddings()[:, 0], True)
        df.insert(1, "Reduced Feature 2", self.training_data_set.get_reduced_embeddings()[:, 1], True)
        df.insert(2, "sentence", self.training_data_set.get_sentences(), True)
        df.insert(3, "label", self.training_data_set.get_label_list(), True)
        
        fig = px.scatter(df,
                         x="Reduced Feature 1", 
                         y="Reduced Feature 2", 
                        #  hover_name="sentence",
                         hover_name=df["sentence"].str.wrap(30).apply(lambda x: x.replace('\n', '<br>')),
                         color="label",
                         hover_data={'label': False, 
                                     'Reduced Feature 1': False,
                                     'Reduced Feature 2': False})
        
        return fig
         
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def display_training_results(self) -> None:

        print(f'slope = {-1*self.logreg_classifier.coef_[0][0]/self.logreg_classifier.coef_[0][1]}')
        print(f'y-int = {-1*self.logreg_classifier.intercept_[0]/self.logreg_classifier.coef_[0][1]}')

        _, ax = plt.subplots(figsize=(4, 3))

        DecisionBoundaryDisplay.from_estimator(
            self.logreg_classifier,
            self.training_data_set.get_reduced_embeddings(),
            cmap=plt.cm.Paired,
            ax=ax,
            response_method="predict",
            plot_method="pcolormesh",
            shading="auto",
            xlabel="Feature 1",
            ylabel="Feature 2",
            eps=0.5,
        )

        # Plot also the training points
        plt.scatter(self.training_data_set.get_reduced_embeddings()[:, 0],
                    self.training_data_set.get_reduced_embeddings()[:, 1],
                    c=self.training_data_set.get_label_index_list(),
                    edgecolors="k",
                    cmap=plt.cm.Paired)

        plt.xticks(())
        plt.yticks(())

        # plt.show()
        return plt.gcf()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def classify(   self,
                    sentence : str) -> tuple:
        
        if self.sentence_transformer == None:
            self.sentence_transformer = SentenceTransformer(self.pretrained_transformer_path)
        
        embedding = list(self.sentence_transformer.encode(sentences=[sentence],convert_to_numpy=True))

        reduced_embedding = self.umap_transformer.transform(X=embedding)

        formatted_sample = np.array(reduced_embedding).reshape(1, -1)

        probs = self.logreg_classifier.predict_proba(formatted_sample)

        predicted_class_index = self.logreg_classifier.predict(formatted_sample)
        predicted_class_label = self.training_data_set.get_label_from_index(predicted_class_index)

        return predicted_class_label, probs[0][predicted_class_index]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def test_classifier(self,
                        test_data_path : str,
                        test_label : str,
                        verbose = False) -> dict:

        file = open(test_data_path,'r', encoding='utf-8')
        test_data_set = DataSet(file_stream=file)

        test_data_set.perform_embedding(SentenceTransformer(self.pretrained_transformer_path))
        test_data_set.perform_reduction(self.umap_transformer)

        if not test_data_set.check_label(test_label):
            print(f'WARNING: label <{test_label}> not in data set.')
            return

        total = len(test_data_set.data_list)

        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        for i, test_sample in enumerate(test_data_set.data_list):
            formatted_sample = np.array(test_sample.reduced_encoding).reshape(1, -1)

            probs = self.logreg_classifier.predict_proba(formatted_sample)

            predicted_class_index = self.logreg_classifier.predict(formatted_sample)
            predicted_class_label = test_data_set.get_label_from_index(predicted_class_index)

            actual_class_label = test_sample.label
            actual_class_index = test_data_set.labels[actual_class_label]

            if actual_class_index == predicted_class_index:
                if actual_class_label == test_label:
                    true_pos += 1
                else:
                    true_neg += 1
            else:
                if actual_class_label == test_label:
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

        num_pos = len(test_data_set.get_data_with_label(test_label))
        num_neg = total - num_pos

        prec = true_pos / (true_pos + false_pos)
        accu = (true_pos + true_neg)/ total
        reca = true_pos / (true_pos + false_neg)

        results_dict = {}
        results_dict['True+'] = true_pos
        results_dict['True-'] = true_neg
        results_dict['False+'] = false_pos
        results_dict['False-'] =  false_neg
        results_dict['Precision'] = prec
        results_dict['Accuracy'] = accu
        results_dict['Recall'] = reca
        results_dict['F1-Score'] = (2*reca*prec/(reca+prec))
        results_dict['total'] = total
        results_dict['#+'] = num_pos
        results_dict['#-'] = num_neg
        
        if verbose :
            pprint(results_dict)

        return results_dict

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def save(self,
             output_path : str) -> bool:
        
        # create the output folder if it does not exist
        if_not_exist_create_dir(output_path)
        
        # save the current 'state' of the classifier in a JSON file
        base_classifer_file_path = os.path.join(output_path, BASE_CLASSIFIER_JSON_FILE_NAME)
        
        base_json_dict = {  'name' : self.name,
                            'pretrained_transformer_path' : self.pretrained_transformer_path,
                            'is_initialized' : self.is_initialized}
        
        with open(base_classifer_file_path, "w+") as final:
            json.dump(base_json_dict, final)
        
        # save the training set to a json file
        if self.training_data_set:
            training_data_set_file_path = os.path.join(output_path, TRAINING_DATA_SET_JSON_FILE_NAME)
            self.training_data_set.save_as_json(training_data_set_file_path)
        
        # save umap transformer to pickle file
        umap_pickle_file_path = os.path.join(output_path, UMAP_PICKLE_FILE_NAME)
        try:
            save_umap_transformer(umap_transformer=self.umap_transformer,
                                  file_path=umap_pickle_file_path)
        except:
            # TODO: add logging
            pass
        
        # save logreg classifier to pickle file
        logreg_pickle_file_path = os.path.join(output_path, LOG_REG_PICKLE_FILE_NAME)
        try:
            save_lr_classifier(lr_classifier=self.logreg_classifier,
                               file_path=logreg_pickle_file_path)
        except:
            # TODO: add logging
            pass
        
 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def load(   self,
                input_path : str) -> bool:   
        
        # load base classifier data form JSON file
        base_classifier_json_file_path = os.path.join(input_path, BASE_CLASSIFIER_JSON_FILE_NAME)

        json_dict = json.load(open(base_classifier_json_file_path,'r'))
        self.name = json_dict['name']
        self.pretrained_transformer_path = json_dict['pretrained_transformer_path']
        self.is_initialized = bool(json_dict['is_initialized'] == 'true')
        
        # load trainging set data
        self.training_data_path = os.path.join(input_path, TRAINING_DATA_SET_JSON_FILE_NAME)
        self.training_data_set = DataSet(file_path=self.training_data_path)
        self.training_data_stream = None    
        
        # load log-reg classifier from pickle file 
        lr_classifier_pickle_file_path = os.path.join(input_path, LOG_REG_PICKLE_FILE_NAME)     
        self.logreg_classifier =  pickle.load((open(lr_classifier_pickle_file_path, 'rb')))
        
        # load umap transformer form pickle file
        umap_transformer_pickle_file_path = os.path.join(input_path, UMAP_PICKLE_FILE_NAME)
        self.umap_transformer =  pickle.load((open(umap_transformer_pickle_file_path, 'rb')))
                  
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~