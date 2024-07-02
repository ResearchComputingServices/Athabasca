import os

import plotly_express as px
import plotly.graph_objects as go
import matplotlib as plt
import numpy as np
import pandas as pd

from pprint import pprint

from SentenceClassifier.Classifier import SentenceClassifier
from SentenceClassifier.FineTuner import fine_tune_llm, generate_interactive_plot
from SentenceClassifier.DataSet import DataSet

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TEST_SENTENCE_1 = ("Just like regular encrypted viruses, a polymorphic virus "
                 "infects files with an encrypted copy of itself, which is "
                 "decoded by a decryption module.")
TEST_SENTENCE_2 = ("The invention of the electron microscope in 1931 brought "
                   "the first images of viruses.")
TEST_SENTENCE_3 = ("Some prisioners are allowed to have computers in their cells.")

TEST_SENTENCES = [TEST_SENTENCE_1, TEST_SENTENCE_2, TEST_SENTENCE_3]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def virus_data_training():
    
    TRAINING_DATA_SET_PATH = 'sample_data/virus_labelled_data_training.csv'
    
    # create an instance of a sentence classifier
    classifier_train = SentenceClassifier(  name = 'VirusClassifier',
                                            pretrained_transformer_path='all-MiniLM-L6-v2',
                                            verbose=False)
                      
    classifier_train.set_train_data_path(training_data_path='sample_data/virus_labelled_data_training.csv')
    # classifier_train.set_train_data_stream(open('sample_data/virus_labelled_data_training.csv', 'r'))
    
    classifier_train.train_classifier() 
    
    # Test the save and load methdods
    classifier_train.save(output_path='my-classifier')   
    
    classifier_loaded = SentenceClassifier()
    classifier_loaded.load(input_path='my-classifier')
    
    for sentence in TEST_SENTENCES:
        label, prob = classifier_loaded.classify_sentence(sentence)
        print(f'[{sentence}] --> {label} conf {prob}')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_example_classifier(   output_path='my-fine-tuned-classifier',
                                full_data_set_training_path='sample_data/label_sentence_data_cleaned.csv',
                                fine_tuned_path='fine-tuned-model',
                                pretrained_transformer_path='all-MiniLM-L6-v2'):
    
    print('Building data set...',flush=True)  
    full_data_set = DataSet(file_path=full_data_set_training_path)
    
    
    print('Fine tuning llm...',flush=True)  
    fine_tuned_path = fine_tune_llm(data_set=full_data_set,
                                    base_output_path=output_path,
                                    path_to_pretrained_llm=pretrained_transformer_path,
                                    num_corrections=25)
    
    print('Initializing Classifier...',flush=True)     
    classifier_fine_tuned = SentenceClassifier( name = 'Fine-Tuned-'+pretrained_transformer_path,
                                                pretrained_transformer_path=fine_tuned_path,
                                                verbose=True)
    
    classifier_fine_tuned.add_training_data_set(full_data_set)
    
    print('Training Classifier...',flush=True)  
    classifier_fine_tuned.train_classifier()

    classifier_fine_tuned.save(output_path=output_path) 
    
    print('Saving Classifier...',flush=True)  
    fig = classifier_fine_tuned.generate_interactive_plot()
    fig.show() 
    
    print('Testing Classifier...',flush=True)  
    testing_data_path = 'sample_data/test.csv'
    testing_data_set = DataSet(file_path=testing_data_path)
    
    results = []

    for label in testing_data_set.get_labels():
        result_dict = classifier_fine_tuned._test_classifier(   test_data_set=testing_data_set,
                                                                test_label=label)
        results.append(result_dict)

    pprint(results)
   
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
 
def load_test():
     
    classifier_fine_tuned = SentenceClassifier()
     
    classifier_fine_tuned.load(input_path='my-fine-tuned-classifier') 
    
    # classifier_fine_tuned.display_training_results()

    #testing_data_path = 'sample_data/test.csv'
    testing_data_path = 'sample_data/test_full.csv'
    testing_data_set = DataSet(file_path=testing_data_path)
        
    testing_data_set.perform_embedding(classifier_fine_tuned._get_sentence_transformer())
    testing_data_set.perform_reduction(classifier_fine_tuned.umap_transformer)
    testing_data_set.normalize_data(classifier_fine_tuned.min_max_scaler)
    
    # for label in testing_data_set.get_labels():
    #     data_list = testing_data_set.get_data_with_label(label)
            
    df = pd.DataFrame()
    df.insert(0, "Reduced Feature 1", testing_data_set.get_reduced_embeddings()[:, 0], True)
    df.insert(1, "Reduced Feature 2", testing_data_set.get_reduced_embeddings()[:, 1], True)
    df.insert(2, "label", testing_data_set.get_label_index_list(), True)
    df.insert(3, "sentence",testing_data_set.get_sentences(), True)
            
    fig = px.scatter(   df,
                        x="Reduced Feature 1", 
                        y="Reduced Feature 2", 
                        hover_name=df["sentence"].str.wrap(30).apply(lambda x: x.replace('\n', '<br>')),
                        color="label",
                        hover_data={'label': False, 
                                    'Reduced Feature 1': False,
                                    'Reduced Feature 2': False})
    
    coeff = classifier_fine_tuned.logreg_classifier.coef_
    inter = classifier_fine_tuned.logreg_classifier.intercept_
    
    print(coeff)
    print(inter)
    
    for i in range(0,3):
        slope = -1*coeff[i][0]/coeff[i][1]
        y_int = -1*inter[i]/coeff[i][1]
        print(f'slope: {slope}, y-int: {y_int}')
        db_x = np.linspace(0, 1, 100)
        db_y = slope*db_x + y_int
        fig.add_trace(go.Line(x=db_x, y=db_y))
    
    fig.update_xaxes(range=[0,1])
    fig.update_yaxes(range=[0,1])
    fig.show()
   
    results = classifier_fine_tuned._test_classifier(test_data_set=testing_data_set, 
                                                     test_label='COMP_CON')
    
    pprint(results)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():

    # #pretrained_transformer_path = 'all-mpnet-base-v2' 
    # #full_data_set_training_path = 'sample_data/label_sentence_data_FULL.csv'     
    
    # pretrained_transformer_path = 'all-MiniLM-L6-v2'
    # full_data_set_training_path = 'sample_data/label_sentence_data_cleaned.csv'    
    # output_path = 'my-fine-tuned-classifier'
    # fine_tuned_llm_path = 'fine-tuned-model'
    
    # train_example_classifier(   output_path=output_path,
    #                             full_data_set_training_path=full_data_set_training_path,
    #                             fine_tuned_path=fine_tuned_llm_path,
    #                             pretrained_transformer_path=pretrained_transformer_path)
    
    load_test()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    main()
