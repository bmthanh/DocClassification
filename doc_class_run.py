# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:26:53 2018

@author: thanh.bui
"""

import numpy as np

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

import doc_class_utilities as doc_utils

# Load the trained model
with open ('training_data_model.pickle', 'rb') as handle:
    rf_model, df_refined = pickle.load(handle)

#%% 
# Extract scientific document features
# The paths are used for training the classifier
#text_paths = ["D:\\Work\\TextAnalysis\\DocClassification\\data\\RandomSelectedDocs\\NSD", 
#            "D:\\Work\\TextAnalysis\\DocClassification\\data\\RandomSelectedDocs\\SD"]

#
# The paths used for testing
text_paths = ['T:\\Projets\\Externes']

text_keys = df_refined.columns.values[:-2]  # An array of keywords for recognizing the document
values = [0]*len(text_keys)
text_key_dict_init_temp = dict(zip(text_keys, values))
text_key_dict_init = OrderedDict(sorted(text_key_dict_init_temp.items()))

no_scanned_pdf = 0      # Count the number of scanned pdf file
verbose = 1

for sci in range(len(text_paths)):
    # Collect the paths of pdf files in the directory
    pdf_file_path = doc_utils.collect_file_path(text_paths[sci], '.pdf') 
    #for i in range(len(pdf_file_path)):
    for i in range(15):   
        # Extract the features
        try:
            #Read the first page of a pdf file and extract text
            (extracted_text, no_pages) = doc_utils.read_pdf_file(pdf_file_path[i], pages=[0])  
            print('===========Index = {}=================='.format(i))
            print(pdf_file_path[i])            
        except:
            continue
        text_key_dict, scanned_pdf = doc_utils.feature_extraction(extracted_text, text_key_dict_init,verbose)
        
        if scanned_pdf:
            no_scanned_pdf += 1
            print('This is a scanned pdf file, expect native pdf')
        else:
            temp = list(text_key_dict.values())
            temp.append(no_pages)
            feature = np.array(temp)
            
        #### Predict the label: sci or nonsci
            label = rf_model.predict(feature.reshape(1,-1))
            if(label==1):
                print('Scientific document')
            else:
                print('Non scientific document')
            
        
