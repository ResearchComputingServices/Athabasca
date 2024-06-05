import time
import csv

import plotly_express as px
from pprint import pprint

from SentenceClassifier.Classifier import SentenceClassifier
from SentenceClassifier.FineTuner import fine_tune_llm, generate_interactive_plot

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

def fine_tune_time_trial():
    
    training_data_path = 'sample_data/label_sentence_data_cleaned.csv'
    pretrained_transformer_path = 'all-MiniLM-L6-v2'
    
    # print('Fine tuning LLM...',end='',flush=True)
    # fine_tuned_path = fine_tune_llm(path_to_data_set=training_data_path,
    #                                 path_to_pretrained_llm=pretrained_transformer_path)
    # print('DONE')   
    fine_tuned_path = pretrained_transformer_path
    
    classifier_fine_tuned = SentenceClassifier( name = 'Fine Tuned',
                                                pretrained_transformer_path=fine_tuned_path,
                                                verbose=False)


    classifier_fine_tuned.set_train_data_path(training_data_path=training_data_path)
    
    classifier_fine_tuned.initialize()

    print('Training Classifier...',end='',flush=True)
    classifier_fine_tuned.train_classifier()
    print('DONE')

    print('Testing Classifer...',end='',flush=True)
    tic = time.time()
    classifier_fine_tuned.test_classifier(  test_data_path='sample_data/test.csv',
                                            test_label='COMP_CON')
    toc = time.time()
    print(f'DONE: {toc-tic}s')
    
    test_data = list(csv.reader(open('sample_data/test.csv', 'r')))
    tic = time.time()
    for test in test_data:
        pprint(test)
        classifier_fine_tuned.classify_sentence(test[1])
    toc = time.time()
    print(f'DONE: {toc-tic}s')
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
        
    # create an instance of a sentence classifier
    classifier_train = SentenceClassifier(  name = 'VirusClassifier',
                                            pretrained_transformer_path='all-MiniLM-L6-v2',
                                            verbose=False)
                        
    classifier_train.set_train_data_path(training_data_path='sample_data/virus_labelled_data_training.csv')
    
    classifier_train.initialize()
    
    classifier_train.train_classifier()
    
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
    fine_tune_time_trial()