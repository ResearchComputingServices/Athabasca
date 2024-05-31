from SentenceClassifier.Classifier import SentenceClassifier

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TEST_SENTENCE_1 = ("Just like regular encrypted viruses, a polymorphic virus "
                 "infects files with an encrypted copy of itself, which is "
                 "decoded by a decryption module.")
TEST_SENTENCE_2 = ("The invention of the electron microscope in 1931 brought "
                   "the first images of viruses.")
TEST_SENTENCE_3 = ("Some prisioners are allowed to have computers in their cells.")

TEST_SENTENCES = [TEST_SENTENCE_1, TEST_SENTENCE_2, TEST_SENTENCE_3]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
        
    # create an instance of a sentence classifier
    classifier_train = SentenceClassifier(  name = 'VirusClassifier',
                                            pretrained_transformer_path='all-MiniLM-L6-v2',
                                            verbose=False)
                        
    classifier_train.set_train_data_path(training_data_path='sample_data/virus_labelled_data_training.csv',)
    
    classifier_train.initialize()
    
    classifier_train.train_classifier()
    
    classifier_train.save(output_path='my-classifier')
    
    classifier_loaded = SentenceClassifier()
    classifier_loaded.load(input_path='my-classifier')
    
    for sentence in TEST_SENTENCES:
        label, prob = classifier_loaded.classify(sentence)
        print(f'[{sentence}] --> {label} conf {prob}')
        
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    main()