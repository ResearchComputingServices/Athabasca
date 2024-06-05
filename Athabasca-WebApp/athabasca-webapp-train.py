import os
from datetime import datetime

import streamlit as st
#from streamlit_image_coordinates import streamlit_image_coordinates

import pandas as pd
import glob
from pprint import pprint
from PIL import Image

from SentenceClassifier.Classifier import SentenceClassifier
from ExDocGen.ExtractedDocumentGenerator import ExtractedDocumentGenerator

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global Constants
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LOCAL_DATA_SET_DIR_PATH = '.local_data_sets'
OUTPUT_DIR_PATH = '.output'

DOC_GEN_KEY = 'DOC_GEN_KEY'
EXTRACTED_DOC_KEY = 'EXTRACTED_DOC_KEY'
DATA_SET_FILE_KEY = 'DATA_SET_FILE_KEY'
DATA__FILE_NAME_KEY = 'DATA__FILE_NAME_KEY'
LOCAL_DATA_SET_LIST_KEY = 'LOCAL_DATA_SET_LIST_KEY'
CLASSIFIER_NAME_KEY = 'CLASSIFIER_NAME_KEY'
CLASSIFIER_KEY = 'CLASSIFIER_KEY'
TRAIN_TEST_RESULTS_KEY = 'TRAIN_TEST_RESULTS_KEY'
TRAIN_FIGURE_KEY = 'TRAIN_FIGURE_KEY'
VIEW_DATA_SET_KEY = 'VIEW_DATA_SET_KEY'
TRAINING_FRACTION_KEY = 'TRAINING_FRACTION_KEY'
SELECTED_PRETRAINED_MODEL_KEY = 'SELECTED_PRETRAINED_MODEL_KEY'
IMAGE_INDEX_MAX_KEY = 'IMAGE_INDEX_MAX_KEY'
IMAGE_FILE_PATHS_KEY = 'IMAGE_FILE_PATHS_KEY'
IMAGE_INDEX_CUR_KEY = 'IMAGE_INDEX_CUR_KEY'
ANALYSIS_COMPLETE_KEY = 'ANALYSIS_COMPLETE_KEY'

if DATA_SET_FILE_KEY not in st.session_state:
    st.session_state[DATA_SET_FILE_KEY] = None

if LOCAL_DATA_SET_LIST_KEY not in st.session_state:
    st.session_state[LOCAL_DATA_SET_LIST_KEY] = []

if VIEW_DATA_SET_KEY not in st.session_state:
    st.session_state[VIEW_DATA_SET_KEY] = False

if CLASSIFIER_NAME_KEY  not in st.session_state:
    st.session_state[CLASSIFIER_NAME_KEY] = None
    
if TRAINING_FRACTION_KEY not in st.session_state:
    st.session_state[TRAINING_FRACTION_KEY] = 0.7
    
if SELECTED_PRETRAINED_MODEL_KEY not in st.session_state:
        st.session_state[SELECTED_PRETRAINED_MODEL_KEY] = None

if DATA__FILE_NAME_KEY not in st.session_state:
    st.session_state[DATA__FILE_NAME_KEY] = None

if CLASSIFIER_KEY not in st.session_state:
    st.session_state[CLASSIFIER_KEY] = None

if TRAIN_TEST_RESULTS_KEY not in st.session_state:
    st.session_state[TRAIN_TEST_RESULTS_KEY] = None

if TRAIN_FIGURE_KEY not in st.session_state:
    st.session_state[TRAIN_FIGURE_KEY] = None

if DOC_GEN_KEY not in st.session_state:
    st.session_state[DOC_GEN_KEY] = ExtractedDocumentGenerator(output_path=OUTPUT_DIR_PATH)

if EXTRACTED_DOC_KEY not in st.session_state:
    st.session_state[EXTRACTED_DOC_KEY] = None

if IMAGE_INDEX_MAX_KEY not in st.session_state:
    st.session_state[IMAGE_INDEX_MAX_KEY] = 0

if IMAGE_FILE_PATHS_KEY not in st.session_state:
    st.session_state[IMAGE_FILE_PATHS_KEY] = None

if IMAGE_INDEX_CUR_KEY not in st.session_state:
    st.session_state[IMAGE_INDEX_CUR_KEY] = 0

if ANALYSIS_COMPLETE_KEY not in st.session_state:
    st.session_state[ANALYSIS_COMPLETE_KEY] = False

# if  not in st.session_state:
#     st.session_state[] =

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Widget Callback Functions 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_classifier(col_fig, col_table):
    
    c = SentenceClassifier(name = st.session_state[CLASSIFIER_NAME_KEY],
                           pretrained_transformer_path='all-MiniLM-L6-v2',
                           verbose=True)

    c.set_train_data_stream(st.session_state[DATA_SET_FILE_KEY])

    c.initialize()
    
    c.train_classifier()
    
    training_results_figure = c.generate_interactive_plot()
   
    testing_results = c.test_classifier(test_data_path = 'test.csv',
                                        test_label =  'COMP_CON',
                                        verbose = False)   
    st.session_state[CLASSIFIER_KEY] = c
    st.session_state[TRAIN_TEST_RESULTS_KEY] = testing_results
    st.session_state[TRAIN_FIGURE_KEY] = training_results_figure
            
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_classifier():
    pass   

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def down_classifier():
    pass   

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def clear_state_data():
    pass   

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def file_change():
    st.session_state[ANALYSIS_COMPLETE_KEY] = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def increase_current_page():
    if st.session_state[IMAGE_INDEX_CUR_KEY] < st.session_state[IMAGE_INDEX_MAX_KEY]:
        st.session_state[IMAGE_INDEX_CUR_KEY] = st.session_state[IMAGE_INDEX_CUR_KEY] + 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

def decrease_current_page():
    if st.session_state[IMAGE_INDEX_CUR_KEY] > 0:
        st.session_state[IMAGE_INDEX_CUR_KEY] = st.session_state[IMAGE_INDEX_CUR_KEY] - 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def view_data_set():
    if st.session_state[DATA_SET_FILE_KEY] != None:
        st.session_state[VIEW_DATA_SET_KEY] = not st.session_state[VIEW_DATA_SET_KEY]
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Other helper function go here

def default_classifier_name() -> str:
    now = datetime.now()
    return 'sent_clsfr_'+str(now.strftime("%m_%d_%Y_%H_%M_%S"))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# Page Functions
def load_data_page():
    
    local_filenames = os.listdir(LOCAL_DATA_SET_DIR_PATH)
    
    # Title
    st.markdown("## Select Data Set")
      
    # Selection Box
    selection_col, _ = st.columns([1,2])
    selected_filename = selection_col.selectbox('Select local file', local_filenames)
    st.session_state[DATA__FILE_NAME_KEY] = selected_filename
    select_file_path = os.path.join(LOCAL_DATA_SET_DIR_PATH, selected_filename)
    
    st.session_state[DATA_SET_FILE_KEY] = open(select_file_path,'r', encoding='utf-8')
    
    selection_col.button(   'View Data Set', 
                            on_click=view_data_set)

    if st.session_state[VIEW_DATA_SET_KEY]:
        st.dataframe(pd.read_csv( st.session_state[DATA_SET_FILE_KEY]),
                                                    use_container_width=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def model_selection_page():    
    
    st.markdown('## Select Initial Pre-Trained LLM')
    st.markdown('give bit of information here on the options')
    
    pretrain_model_options = ['all-MiniLM-L6-v2',
                              'nli-distilroberta-base-v2',
                              'paraphrase-MiniLM-L6-v2']
    
    user_selected_model = st.radio('Select Pre-Trained Model', 
                                   options=pretrain_model_options )
    
    st.session_state[SELECTED_PRETRAINED_MODEL_KEY] = user_selected_model
 
 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_classifier_page():
    # TODO: Add check to see if data is available
    st.markdown('## Train / Test Sentence Classifier')

    controls_col, _ = st.columns([1,2])

    default_name = default_classifier_name()

    if st.session_state[CLASSIFIER_KEY] != None:
        default_name = st.session_state[CLASSIFIER_KEY].name

    class_name = controls_col.text_input(   label='Classifier Name', 
                                            value=default_name)
    
    st.session_state[CLASSIFIER_NAME_KEY] = class_name
    
    col_fig, col_table = st.columns([1,1])
    
    controls_col.slider(label='Select Training/Testing Fraction',
                        min_value=0.6,
                        max_value=1.0,
                        key = TRAINING_FRACTION_KEY,
                        value = 0.7)
   
    controls_col.button('Train', 
                        on_click=train_classifier,
                        args=(col_fig,col_table),)
    
    if st.session_state[TRAIN_TEST_RESULTS_KEY] != None:
        col_table.table(st.session_state[TRAIN_TEST_RESULTS_KEY])
    
    if st.session_state[TRAIN_FIGURE_KEY] != None:
        col_fig.plotly_chart(st.session_state[TRAIN_FIGURE_KEY])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def perform_classifier_page():
    
    if st.session_state[CLASSIFIER_KEY] == None:
        st.error('No Classifier trained.')
        return
    
    col_uploader, col_bar = st.columns([1,2])
    
    uploaded_file = col_uploader.file_uploader("Choose a file", on_change=file_change)
    
    prev, next, _ = st.columns([1,1,15])
    prev.button('Prev',on_click=decrease_current_page)
    next.button('Next',on_click=increase_current_page)
    st.write(f'Current Page { st.session_state[IMAGE_INDEX_CUR_KEY]} of { st.session_state[IMAGE_INDEX_MAX_KEY]}')
    col_pdf, col_text = st.columns(2)
   
    if uploaded_file != None and not st.session_state[ANALYSIS_COMPLETE_KEY]:
        # get the document extractor
        doc_gen = st.session_state[DOC_GEN_KEY]
        st.session_state[EXTRACTED_DOC_KEY] = doc_gen.extract_from_stream(  uploaded_file.getvalue(),
                                                                            output_name='test')
        st.session_state[IMAGE_FILE_PATHS_KEY] = glob.glob(OUTPUT_DIR_PATH+'/annotated_images/*.png')
        st.session_state[IMAGE_FILE_PATHS_KEY].sort()
                
        st.session_state[IMAGE_INDEX_MAX_KEY] = len(st.session_state[IMAGE_FILE_PATHS_KEY])-1

        # get the sentence classifier object
        c = st.session_state[CLASSIFIER_KEY]

        prog_bar = col_bar.progress(0, text='Working...')

        total_num_blocks = st.session_state[EXTRACTED_DOC_KEY].num_text_blocks
        blocks_complete = 0
        for page in st.session_state[EXTRACTED_DOC_KEY]:
            for text_block in page:
                for sentence in text_block.sentences:
                    sentence.label, sentence.conf = c.classify(str(sentence))
                blocks_complete += 1
                prog_bar.progress(blocks_complete/total_num_blocks, text='Working...')
        st.session_state[ANALYSIS_COMPLETE_KEY] = True
                                    
    col_pdf.markdown("## PDF Images")
    if st.session_state[EXTRACTED_DOC_KEY] != None:
        current_page = st.session_state[IMAGE_INDEX_CUR_KEY]
               
        image_file_path = st.session_state[IMAGE_FILE_PATHS_KEY][current_page]
        
        image = Image.open(image_file_path)
    
        # this gets the location in image coordinates where the click has
        # happened on the image.
        with col_pdf:
            value =  streamlit_image_coordinates(image)  
            print(value)
    
    col_text.markdown("## Extracted Text")
    
    if st.session_state[EXTRACTED_DOC_KEY] != None:  
        extracted_document = st.session_state[EXTRACTED_DOC_KEY]
        current_page = extracted_document.get_page(st.session_state[IMAGE_INDEX_CUR_KEY])
                
        col_text.markdown(current_page.get_labelled_text_full())
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def home_page():
    st.title("Athabasca ")
    st.markdown("Sematic sentence classifier using logistics regression and LLM generated sentence embeddings")
    st.markdown(" [GitHub Repo](https://github.com/ResearchComputingServices/Athabasca) ")
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# initialize the page
st.set_page_config(page_title="Sematic Sentence Classifier",
                   layout="wide")

st.sidebar.title('Navigation')
user_page_selection = st.sidebar.radio('Pages', options = ['Home', 'Load Data', 'Select LLM', 'Train / Test', 'Classify'])

if user_page_selection == 'Load Data':
    load_data_page()
elif user_page_selection == 'Select LLM':
    model_selection_page()
elif user_page_selection == 'Train / Test':
   train_classifier_page()
elif user_page_selection == 'Classify':
    perform_classifier_page()
else:
    home_page()

st.sidebar.button('Save', on_click=save_classifier)
st.sidebar.button('Download', on_click=down_classifier)
st.sidebar.button('Clear', on_click=clear_state_data)

file_name = 'None'
if st.session_state[DATA__FILE_NAME_KEY] != None:
    file_name = st.session_state[DATA__FILE_NAME_KEY]
st.sidebar.markdown(f'Data Set: {file_name}')

model_name = 'None'
if st.session_state[SELECTED_PRETRAINED_MODEL_KEY] != None:
    model_name = st.session_state[SELECTED_PRETRAINED_MODEL_KEY]
st.sidebar.markdown(f'LLM: {model_name}')

classifer_name = 'None'
if st.session_state[CLASSIFIER_NAME_KEY] != None:
    classifer_name = st.session_state[CLASSIFIER_NAME_KEY]
st.sidebar.markdown(f'Classifier: {classifer_name}')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




