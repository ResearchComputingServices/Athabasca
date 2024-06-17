import time
import csv

import plotly_express as px
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

def run_classifier(classifier : SentenceClassifier,
                   train_data_path : str):
    classifier.set_train_data_path(training_data_path=train_data_path)
    classifier.initialize()
    
    return generate_interactive_plot(classifier.training_data_set)

def fine_tune_transformer_comparison():
 
    training_data_path = 'sample_data/label_sentence_data_cleaned.csv'
    pretrained_transformer_path = 'all-MiniLM-L6-v2'
    
    fine_tuned_path = fine_tune_llm(path_to_data_set=training_data_path,
                                    path_to_pretrained_llm=pretrained_transformer_path)
    
    classifier_pre_train = SentenceClassifier(  name = 'Pre Trained',
                                                pretrained_transformer_path='all-MiniLM-L6-v2',
                                                verbose=False)
    
    classifier_fine_tuned = SentenceClassifier( name = 'Fine Tuned',
                                                pretrained_transformer_path=fine_tuned_path,
                                                verbose=False)
    
    pre_trained_fig = run_classifier(classifier_pre_train,training_data_path)    
    fine_tuned_fig =  run_classifier(classifier_fine_tuned,training_data_path)    
    
    pre_trained_fig.show()
    fine_tuned_fig.show()
           
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

def fine_tune_test():
    
    pretrained_transformer_path = 'all-MiniLM-L6-v2'
    
    # testing_data_path = 'sample_data/test.csv'
    # training_data_path = 'sample_data/label_sentence_data_cleaned.csv'
    
    #training_data_path = 'sample_data/label_sentence_data_balanced.csv'
    
    #testing_data_path = 'sample_data/test_full.csv'
    #training_data_path = 'sample_data/label_sentence_data_FULL.csv'
    
    full_data_set_training_path = 'sample_data/label_sentence_data_cleaned.csv'    
    full_data_set = DataSet(file_path=full_data_set_training_path)
    
    train_set, test_set = full_data_set.split_training_testing(0.7)
        
    fine_tuned_path = fine_tune_llm(data_set=train_set,
                                    path_to_pretrained_llm=pretrained_transformer_path,
                                    num_corrections=5)
      
    classifier_fine_tuned = SentenceClassifier( name = 'Fine Tuned',
                                                pretrained_transformer_path=fine_tuned_path,
                                                verbose=False)
    
    classifier_fine_tuned.add_data_set(train_set)
    
    classifier_fine_tuned.train_classifier()

    classifier_fine_tuned.save(output_path='my-fine-tuned-classifier') 

    classifier_load = SentenceClassifier()
    classifier_load.load(input_path='my-fine-tuned-classifier')

    results = []

    for label in test_set.get_labels():
        result_dict = classifier_load._test_classifier( test_data_set=test_set,
                                                        test_label=label)
        results.append(result_dict)

    fig = classifier_load.generate_interactive_plot()

    pprint(results)
    
    fig.show()   
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    
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

if __name__ == '__main__':
    #main()
    #fine_tune_transformer_comparison()
    fine_tune_test()