from Athabasca import SentenceClassifier

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TEST_SENTENCE_1 = ("Just like regular encrypted viruses, a polymorphic virus "
                 "infects files with an encrypted copy of itself, which is "
                 "decoded by a decryption module.")
TEST_SENTENCE_2 = ("The invention of the electron microscope in 1931 brought "
                   "the first images of viruses.")

TEST_SENTENCES = [TEST_SENTENCE_1, TEST_SENTENCE_2]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
        
    # create an instance of a sentence classifier
    c = SentenceClassifier(name = 'VirusClassifier',
                           training_data_path='sample_data/virus_labelled_data_training.csv',
                           pretrained_transformer_path='all-MiniLM-L6-v2',
                           verbose=False)
    
    c.initialize()
    
    c.train_classifier()
    
    for sentence in TEST_SENTENCES:
    
        label, prob = c.classify(sentence)
        print(f'[{sentence}] --> {label} conf {prob}')
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    main()