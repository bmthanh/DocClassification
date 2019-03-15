# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:20:28 2018

@author: thanh.bui
"""
import os
import numpy as np
import shutil
import io

#%%
def extract_files(src_path, dest_path):
    '''Find documents with specified extentions from scr_path and then copy them to dest_path
    '''
    ext = [".pdf", ".doc", ".docx", ".tiff"]
    file_paths = list()
    file_names = list()
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith(tuple(ext)):  # Check the file extension
                #print(file)
                s = os.path.join(root, file)
                if(os.path.exists(s) and file[0] != "~"):
                    file_paths.append(s)
                    file_names.append(file)
                
    
    NF = len(file_paths)
    print("Number of files found:", NF)
    np.random.seed(1)
    percentage = 100
    print("...Copying", int(NF*percentage/100), "files, please wait.........")
    for i in np.random.randint(0, NF-1, size = int(NF*percentage/100)):
        if(os.path.exists(os.path.join(dest_path,file_names[i])) == False):
            if(os.path.exists(file_paths[i])):
                shutil.copy2(file_paths[i], dest_path)
    return True

#%%
def collect_file_path(path, ext):
    '''Collect file paths of the files with the specified extension
    '''
    file_paths = list()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                file_paths.append(os.path.join(root, file))
    return file_paths

#%% Read pdf files and extract text
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import resolve1

def read_pdf_file(fname, pages=None):
    '''Read a pdf file and return the text of specified page and number of pages of the document
    '''
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = io.StringIO()
    codec = 'utf-8'
    imagewriter = None
    manager = PDFResourceManager()
    laparams=LAParams()
    laparams.all_texts = True  # Correspond to -A option
    device = TextConverter(manager, output,  codec=codec, laparams=laparams,
                           imagewriter=imagewriter)
    #device = TextConverter(manager, output, laparams=laparams)
    interpreter = PDFPageInterpreter(manager, device)
    infile = open(fname, 'rb')
    
    # Read the number of pages
    parser = PDFParser(infile)
    document = PDFDocument(parser)
    no_pages = resolve1(document.catalog['Pages'])['Count']
    
    # Inteprete the text of the document
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
        
    infile.close()
    device.close()
    text = output.getvalue()
    output.close
    return text, no_pages
#%% Feature extraction from text
def feature_extraction(extracted_text, text_key_dict_init, verbose=1):
    '''Extract features from text
    
    '''
    text_key_dict = text_key_dict_init.copy()
    extracted_text_r = extracted_text.replace('\n', ' ') # remove '\n' character in the middle of string
    extracted_text_low = extracted_text_r.lower()
    tokens = word_tokenize(extracted_text_low)
    scanned_pdf = 0
    
    if tokens: # tokens is not empty
        punctuations = ['(', ')', ';', ':', '[', ']', ',', '.', '“', '”']
        tokens = [word for word in tokens if not word in punctuations]
        
        # Check if tokens is just in characters, or the word is split in characters
        if (len(tokens) > 100):
            rand_num_size = int(len(tokens)/30)
            rand_num = np.random.randint(20, size=rand_num_size)
        else:
            rand_num_size = len(tokens)
            rand_num = range(rand_num_size)
        len_tokens = 0
        for i in rand_num:
            len_tokens += len(tokens[i])
        if(len_tokens == rand_num_size): #tokens are just characters
            print('Recompute the tokens ...')
            extracted_text_r = extracted_text.replace('\n', '') # remove '\n' character in the middle of a string
            extracted_text_low = extracted_text_r.lower()
            tokens = word_tokenize(extracted_text_low)    
        
        # Tokens contain words
        from collections import Counter  
        count = Counter(tokens) 
        if(verbose):
            print(count.most_common(20))
        for key in text_key_dict_init.keys():
            for count_key, count_value in count.items():
                if key == count_key:
                    text_key_dict[key] = count_value
                    if(verbose):
                        print(count_key, count_value)
    else:
        scanned_pdf = 1
        print('Tokens is empty, scanned pdf is read')  
    if(verbose):
        print(text_key_dict)
    return text_key_dict, scanned_pdf

#%%
def report_grid_search(results, n_top=3):
    '''Report the first three best parameter sets from grid search results
    '''
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")   
    