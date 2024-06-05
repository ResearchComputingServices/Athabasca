import json
import csv
import random
import re
import logging
import _io

from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn import preprocessing as p

import numpy as np

# =========================================================================

class Datum:
    """Object represents a single sentence, it's label and encodings
    """

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, 
                 sentence = '',
                 label= '',
                 encoding = [],
                 reduced_encoding = []):
        
        self.sentence = sentence
        self.label = label
        self.encoding = [float(x) for x in encoding]
        self.reduced_encoding = [float(x) for x in reduced_encoding]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def to_dict(self) -> dict:
        """returns the contains of the object in dictionary form

        Returns:
            dict: dictionary containing all the data of the object
        """
        return {'sentence' : self.sentence,
                'label' : self.label,
                'encoding' : [str(x) for x in self.encoding],
                'reduced_encoding' : [str(x) for x in self.reduced_encoding]}


# =========================================================================

class DataSet:
    """ This object is a set of labelled sentences and their encodings
    """

    def __init__(self, *,
                 file_path = None,
                 file_stream = None,
                 sentence_list = None):
   
        self.data_list = []
        self.labels = {}

        self.is_embedded = False
        self.is_reduced = False

        self.file_path = ''
        if file_path != None:
            self.file_path = file_path
            if file_path.endswith('.json'):
                self.load_json(file_path)
            elif file_path.endswith('.csv'):
                self.load_csv(file_path)           
        elif file_stream != None:
            self.load_stream(file_stream)
        elif sentence_list != None:
            self.read_list(sentence_list)
            
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @property
    def n_samples(self) -> int:
        return len(self.data_list)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def check_label(self,
                    test_label : str) -> bool:
        """Checks if test_label is contained in the data set

        Args:
            test_label (str): query label

        Returns:
            bool: True if test_label in data set
        """
        
        return test_label in self.labels.keys()           

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_sentences(self) -> list:
        """returns a list of all sentences in DataSet

        Returns:
            list: of all sentences in DataSet
        """
        return [d.sentence for d in self.data_list]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_embeddings(self) -> np.ndarray:
        """returns full embeddings in a numpy array

        Returns:
            np.ndarray: embeddings in a numpy array
        """
        
        embeddings_list = []
        
        if self.is_embedded:
            embeddings_list = [d.encoding for d in self.data_list]
        else:
            logging.warning('Attempted to get embeddings from data set, but none available!')
        
        return np.array(embeddings_list)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_reduced_embeddings(self) -> np.ndarray:
        """returns all the reduced embeddings in a numpy array

        Returns:
            np.ndarray: reduced embeddings in a numpy array
        """
        
        reduced_embeddings_list = []
        
        if self.is_reduced:
            reduced_embeddings_list = [d.reduced_encoding for d in self.data_list]
        
        return np.array(reduced_embeddings_list)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_label_index_list(self) -> np.ndarray:
        """returns a numpy array containing a numerical index for the label
        of each sentence in the data set

        Returns:
            np.ndarray: label indices in a numpy array
        """
        return np.array([self.labels[d.label] for d in self.data_list])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
    def get_label_list(self) -> list:
        """returns a numpy array containing strings for the label
        of each sentence in the data set

        Returns:
            list: label string in a list
        """
        return [d.label for d in self.data_list]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_data_with_label(self,
                            arg_label : str) -> list:
        """returns a list of all datums with label equal to arg_label

        Args:
            arg_label (str): label

        Returns:
            list: list of datums
        """
        return [d for d in self.data_list if d.label == arg_label]
         
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def split_training_testing(self,
                               training_fraction = 0.9) -> tuple:

        """Spilts the data set into two parts

        Returns:
            tuple: returns two DataSets one for training and one for testing
        """
        random.shuffle(self.data_list)

        ds_training = DataSet()
        ds_testing  = DataSet()

        split_index = int(len(self.data_list)*training_fraction)

        ds_training.labels = self.labels
        ds_testing.labels = self.labels

        ds_training.data_list = self.data_list[:split_index]
        ds_testing.data_list = self.data_list[split_index:]

        ds_training.is_embedded = self.is_embedded
        ds_testing.is_embedded = self.is_embedded

        return ds_training, ds_testing

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_label_from_index(self, 
                             index : int) -> str:
        """returns the string label corresponding to the arg index

        Args:
            index (int): label indice

        Returns:
            str: label string
        """
        for item in self.labels.items():
            if item[1] == index:
                return item[0]
            
        return 'UNKNOWN'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def save_as_json(self, 
                     file_path : str) -> None:
        """Save the DataSet as a JSON file

        Args:
            file_path (str): File path to save DataSet
        """
        with open(file_path, "w+") as final:
            json.dump(self._get_json_dict(), final)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        

    def load_stream(self,
                    file_stream : _io.BufferedReader) -> None:

        reader = csv.reader(file_stream)
        data = list(reader)
                
        # item = (label,sentence)
        for item in data:
            # Save the label into the dict of labels if it is not already there
            if item[0] not in self.labels.keys():
                self.labels[item[0]] = len(self.labels)

            # Add datum to data set
            self.data_list.append(Datum(sentence=re.sub(r'[^\x00-\x7F]+',' ', item[1]),
                                        label=item[0]))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        

    def read_list(  self,
                    sentences : list) -> None:
        
        for sentence in sentences:
            datum = Datum(sentence=re.sub(r'[^\x00-\x7F]+',' ', sentence),label='UNKNOWN')
            self.data_list.append(datum)  

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        


    def load_json(  self, 
                    file_path : str) -> None:
        """Load the DataSet from a JSON file

        Args:
            file_path (str): Path to JSON file containing the DateSet
        """
        json_dict = json.load(open(file_path,'r'))


        self.is_embedded = json_dict['is_embedded']
        self.is_reduced = json_dict['is_reduced']
        self.file_path = json_dict['file_path']
        self.labels = json_dict['labels']

        for datum in json_dict['data']:
            self.data_list.append(Datum(**datum))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        

    def load_csv(self, 
                 file_path : str) -> None:
        """Load labelled data from a CSV file. Format "label","Sentence"

        Args:
            file_path (str): file path to CSV file.
        """

        self.file_path = file_path

        with open(file_path, 'r', encoding='utf-8') as file:
            self.load_stream(file)
                 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _get_json_dict(self) -> dict:
        """Convert the DataSet into a json dictionary

        Returns:
            dict: JSON dictionary representation of the DateSet
        """
        json_dict = {'file_path' : self.file_path,
                     'labels' : self.labels,
                     'is_reduced' : self.is_reduced,
                     'is_embedded' : self.is_embedded,
                     'data' : []}        

        for datum in self.data_list:
            json_dict['data'].append(datum.to_dict())

        return json_dict

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def perform_embedding(  self,
                            sentence_transformer : SentenceTransformer) -> None:
        """Use sentence_transformed to calculate the embedding for each setnence

        Args:
            sentence_transformer (SentenceTransformer): a sentence transformer
        """
                
        embeddings = sentence_transformer.encode(   sentences=self.get_sentences(), 
                                                    show_progress_bar=True,
                                                    convert_to_numpy=True)
        
        for i, embedding in enumerate(embeddings):
            self.data_list[i].encoding = list(embedding)
            
        self.is_embedded = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_number_of_tokens(self,
                             sentence_transformer : SentenceTransformer) -> None:
        """Display the number of tokens for each sentence in the data set

        Args:
            sentence_transformer (SentenceTransformer): a sentence transformer
        """

        for sentence in self.get_sentences():           
            o = sentence_transformer.tokenizer( sentence, 
                                                return_attention_mask=False, 
                                                return_token_type_ids=False)
            
            num_characters = len(sentence)
            num_words = len(sentence.split(' '))
            num_tokens = len(o.input_ids)
            max_seq_len = sentence_transformer.max_seq_length
            if num_tokens > max_seq_len:
                print(f'{num_characters} / {num_words} / {num_tokens} / {max_seq_len}')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def perform_reduction(self,
                          umap_transformer : UMAP) -> None:
        """Perform dimensional reduction on embeddings

        Args:
            umap_transformer (UMAP): UMAP object
        """
        
        if self.is_embedded:
            reduced_embeddings = umap_transformer.transform(X=self.get_embeddings())
                
            for i, reduced in enumerate(reduced_embeddings):
                self.data_list[i].reduced_encoding = list(reduced)
                
            self.is_reduced = True
        # TODO: add logging to warn the user the embedding needs to be done first
        #else:
        # logging....   
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def normalize_data(self,
                       min_max_scaler : p.MinMaxScaler):
        """Scale reduced embeddings to between 0 and 1

        Args:
            min_max_scaler (p.MinMaxScaler): object of type sklearn.preprocessing.MinMaxScaler
        """

        normalized_data = min_max_scaler.transform(self.get_reduced_embeddings())  

        for i, normalized in enumerate(normalized_data):
            self.data_list[i].reduced_encoding = list(normalized)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def save_reduced_embedding(self,
                               file_path : str):
        """Write reduced embeddings to file

        Args:
            file_path (str): File path where reduced embeddings should be written
        """

        with open(file_path, 'w+', encoding='utf-8') as output_file:
            for datum in self.data_list:
                output_file.write(" ".join([str(i) for i in datum.reduced_encoding])+'\n')

            output_file.write('\n\n')
       
 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           
    
    def save_topic_reduced_embedding(   self,
                                        arg_label : str,
                                        file_path : str):
        """Writes the reduced embeddings of all sentences with label equal to arg_label

        Args:
            arg_label (str): Write sentences with this label
            file_path (str): File path where reduced embeddings should be written
        """

        with open(file_path, 'w+', encoding='utf-8') as output_file:                        
            for datum in self.data_list:
                if datum.label == arg_label:
                    output_file.write("^".join([str(i) for i in datum.reduced_encoding]) + '^' + datum.sentence + '\n')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def clean_data(self,
                   regex_pattern = r'[^a-zA-Z\s]+',
                   min_length = 10) -> None:
        """ Remove none "printable" ascii characters from sentences

        Args:
            regex_pattern (regexp, optional): regex cleaning. Defaults to r'[^a-zA-Z\s]+'.
        """
        cleaned_data_list = []
        
        for datum in self.data_list:
            # TODO: This needs to be fixed so that multiple spaces become single spaces not 0 spaces!
            clean_sentence = re.sub(regex_pattern, "", datum.sentence)
                        
            if len(clean_sentence) > min_length:
                    cleaned_data_list.append(datum)
                
        
        self.data_list = cleaned_data_list
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~