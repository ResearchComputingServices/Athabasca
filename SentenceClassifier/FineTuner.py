import random

import plotly_express as px
import pandas as pd

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from pprint import pprint

from .DataSet import DataSet

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PRE_TRAIN_MODEL = 'all-MiniLM-L6-v2'
FINE_TUNED_MODEL_PATH = './fine-tuned-model'
PERCENT_TEST = 0.1
OUTLIER_JSON_FILE_PATH = 'json/fine_tuning_corrections.json'
MAX_CORRECTIONS = 50

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def generate_interactive_plot(training_data_set : DataSet) -> None:
        
        df = pd.DataFrame()
        df.insert(0, "Reduced Feature 1", training_data_set.get_reduced_embeddings()[:, 0], True)
        df.insert(1, "Reduced Feature 2", training_data_set.get_reduced_embeddings()[:, 1], True)
        df.insert(2, "sentence", training_data_set.get_sentences(), True)
        df.insert(3, "label", training_data_set.get_label_list(), True)
        
        fig = px.scatter(df,
                         x="Reduced Feature 1", 
                         y="Reduced Feature 2", 
                         hover_name=df["sentence"].str.wrap(30).apply(lambda x: x.replace('\n', '<br>')),
                         color="label",
                         hover_data={'label': False, 
                                     'Reduced Feature 1': False,
                                     'Reduced Feature 2': False})
        
        return fig   

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def generate_fine_tuning_data(  train_data_file_path : str,
                                max_corrections = MAX_CORRECTIONS,
                                skip_labels = ['Irrelevant']) -> dict:
            
    full_data_set = DataSet(file_path=train_data_file_path)
    
    correction_samples = {'corrections' : []}

    labels = list(full_data_set.labels.keys())

    # loop over all the samples with labels no in the skip_labels list (ex. 'Irrelevant')
    for sample in full_data_set.data_list:
 
        if sample.label in skip_labels:
            continue
        
        # add up to max_corrections instances of corrections for each label type in the dataset
        for label in labels:   
             
            similariyt_score = 0.
            if label == sample.label:
                similariyt_score = 1.
            
            sentences = full_data_set.get_data_with_label(label)
            random.shuffle(sentences)
            
            for counter, sentence in enumerate(sentences):
                correction_samples['corrections'].append({  'sentence 1' : sample.sentence,
                                                            'sentence 2' : sentence.sentence,
                                                            'similarity' : similariyt_score})
                
               
                if counter > max_corrections:
                    break
                            
    return correction_samples
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fine_tune_llm(  path_to_data_set : str,
                    path_to_pretrained_llm = PRE_TRAIN_MODEL):
    
    fine_tuning_data = generate_fine_tuning_data(train_data_file_path=path_to_data_set)
    
    # Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer(path_to_pretrained_llm)
    
    # Define your train examples. You need more than just two examples...
    train_examples = []
    
    for sample in fine_tuning_data['corrections']:
        
        input_example = InputExample(texts=[sample['sentence 1'], sample['sentence 2']], label=float(sample['similarity']))
        
        train_examples.append(input_example)
            
    random.shuffle(train_examples)
    
    TEST_INDEX = int(len(train_examples)*PERCENT_TEST)

    print(f'# of training examples {len(train_examples)}')
    print(f'train: {len(train_examples[TEST_INDEX:])}')
    print(f'test: {len(train_examples[:TEST_INDEX])}')

    sentences1 = []
    sentences2 = []
    scores = []

    for item in train_examples[:TEST_INDEX]:
        sentences1.append(item.texts[0])
        sentences2.append(item.texts[1])
        scores.append(item.label)

    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples[TEST_INDEX:], shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Tune the model
    model.fit(train_objectives=[(   train_dataloader, train_loss)], 
                                    epochs=1, 
                                    warmup_steps=100,
                                    save_best_model=True,
                                    output_path=FINE_TUNED_MODEL_PATH,
                                    show_progress_bar=True,
                                    evaluator=evaluator,
                                    evaluation_steps=500)
    
    return FINE_TUNED_MODEL_PATH