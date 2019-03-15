# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:26:53 2018

@author: thanh.bui
"""

import numpy as np

import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from collections import OrderedDict

import doc_class_utilities as doc_utils

#%% Step 1: Scan files in directory
#src_path = "D:\Work\TextAnalysis\DocClassification\SourceDocs"
#dest_path = "D:\Work\TextAnalysis\DocClassification\SelectedDocs"
#extract_files(src_path, dest_path)

#%% 
# Extract scientific document features
text_paths = ["D:\\Work\\TextAnalysis\\DocClassification\\data\\RandomSelectedDocs\\NSD", 
            "D:\\Work\\TextAnalysis\\DocClassification\\data\\RandomSelectedDocs\\SD"]

text_keys = ['keywords', 'abstract', 'volume+number', 'introduction', 'authors', 'publication', 'index','no.', 'by', 'rev.', 'journal', 'jnl', 'laboratory', 'university', 'univ.', \
            'al.', 'manuscript','tutorial', 'institute', 'research', 'doi', 'published', 'received', 'article', 'conference', 'paper', 'mot-cles', 'resumé', 'par', 'institut', 'rapporteur']
values = [0]*len(text_keys)
text_key_dict_init_temp = dict(zip(text_keys, values))
text_key_dict_init = OrderedDict(sorted(text_key_dict_init_temp.items()))


data = []
no_scanned_pdf = 0      # Count the number of scanned pdf file

for sci in range(len(text_paths)):
    pdf_file_path = doc_utils.collect_file_path(text_paths[sci], '.pdf') # Collect the paths of pdf files in the directory

    for i in range(len(pdf_file_path)):
        try:
            #Read the first page of a pdf file and extract text
            (extracted_text, no_pages) = doc_utils.read_pdf_file(pdf_file_path[i], pages=[0])  
            print('===========Index = {}=================='.format(i))
            print(pdf_file_path[i])            
        except:
            continue
        # Extract the features
        text_key_dict, scanned_pdf = doc_utils.feature_extraction(extracted_text, text_key_dict_init)
        
        if scanned_pdf:
            no_scanned_pdf += 1
        else:
            temp = list(text_key_dict.values())
            temp.extend([no_pages, sci])  #use list.append() to append a single value, and list.extend() to append multiple values.
            data.append(temp)       
        print('No of pages: {}'.format(no_pages))
       
    data_np = np.array(data)
    print('{0} scanned pdf out of {1} pdf files'.format(no_scanned_pdf, len(pdf_file_path)))

#
with open('data_np.pickle', 'wb') as handle:
    pickle.dump([data_np, text_key_dict_init], handle, protocol=pickle.HIGHEST_PROTOCOL)    

#%% Check statistics of data_np

with open('data_np.pickle', 'rb') as handle:
    dataset, text_key_dict_init = pickle.load(handle)   

columns_df = list(text_key_dict_init.keys())
columns_df.extend(['no_pages', 'sci'])
df = pd.DataFrame(dataset, columns = columns_df)
corr_matrix = df.corr()
corr_matrix['sci'].sort_values(ascending=False)

df_refined = df.drop(['research', 'conference','jnl', 'mot-cles', 'no.', 'resumé', 'tutorial', 'univ.', 'volume+number'], axis=1)

df_refined.head()

#% Evaluatation
refine = 1  # 1: use refined data; 0: use extracted data
if refine:
    data = df_refined.values
else:
    data = df.values

X = data[:,:-1]
y = data[:,-1]


# Doing cross validation quickly
svm_cls = SVC (1, kernel='linear')
scores = cross_val_score(svm_cls, X, y, cv = 5)
print(scores)
print('score average: {}'.format(np.mean(scores)))

#%% Dimensionality reduction for visualization
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
X = data[:,:-1]
y = data[:,-1]
pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)
X_tsne = TSNE(n_components=2).fit_transform(X)
# Percentage of variance explained for each components
print('\nExplained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['turquoise', 'darkorange']
labels = ['nonsci', 'sci']
lw = 1
for color, i, label in zip(colors, [0, 1], labels):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.5, lw=lw, label=label)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')


plt.figure()
for color, i, label in zip(colors, [0, 1], labels):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], color=color, alpha=.5, lw=lw, label=label)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('TSNE')

#%% Grid search for finding optimized parameters
        
# SVM classifier            
svc_model = SVC()
svc_parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 10, 100]}

svc_gs = GridSearchCV(svc_model, svc_parameters, cv=5)
svc_gs.fit(X, y)
svc_gs.best_params_
doc_utils.report_grid_search(svc_gs.cv_results_)

# Random forest
rf_model = RandomForestClassifier()
rf_parameters = [ {'n_estimators': [10, 50, 100, 1000], 'max_features': [2, 5, 10, 15]},
               {'bootstrap': [False], 'n_estimators': [10, 50, 100, 1000], 'max_features': [2, 5, 10, 15]}]

rf_gs = GridSearchCV(rf_model, rf_parameters, cv=5)
rf_gs.fit(X,y)
rf_gs.best_params_
doc_utils.report_grid_search(rf_gs.cv_results_)


# GradientBoostingClassifier
gbc_model = GradientBoostingClassifier()
gbc_parameters = {'n_estimators': [10, 100, 1000], 'max_features': [2, 5, 10, 15], 'learning_rate': [0.1, 0.2, 0.5]}

gbc_gs = GridSearchCV(gbc_model, gbc_parameters, cv=5)
gbc_gs.fit(X, y)
gbc_gs.best_params_
doc_utils.report_grid_search(gbc_gs.cv_results_)

with open('gridsearch_result.pickle', 'wb') as handle:
    pickle.dump([svc_gs, rf_gs, gbc_gs], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('gridsearch_result.pickle', 'rb') as handle:
    svc_gs, rf_gs, gbc_gs = pickle.load(handle)

#%% Split the data by preserving the percentage of samples of each class
# Feature scaling
from sklearn.preprocessing import StandardScaler
features = data[:,:-1]    
scaler = StandardScaler()
features_sc = scaler.fit_transform(features)
data_transformed = np.c_[features_sc, data[:,-1]]
scale_feature = 0
if scale_feature:
    data_split = data_transformed
else:
    data_split = data

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_gnb, accuracy_mnb, accuracy_svc = [], [], []
accuracy_rfc, accuracy_gbc = [], []

for i in range(100):
    # Stratified sampling is useful for unbalanced class dataset
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2) #:random_state=42, best: 46
    for train_index, test_index in split.split(data_split[:,:-1], data_split[:,-1]):
        strat_train_set = data_split[train_index]
        strat_test_set = data_split[test_index]
    
    X_train, y_train = strat_train_set[:,:-1], strat_train_set[:,-1]
    X_test, y_test = strat_test_set[:,:-1], strat_test_set[:,-1]
    
    #-------------- Gaussian naive bayes ---------------
    gnb_final = GaussianNB()
    gnb_final.fit(X_train, y_train)
    y_predict_gnb = gnb_final.predict(X_test)
    accuracy_gnb.append(accuracy_score(y_test, y_predict_gnb))
    print('-----Gaussian naive bayes ---------- \nTesting accuracy: ', accuracy_gnb)
    print('Confusion matrix \n', confusion_matrix(y_test, y_predict_gnb))
    
    #-------------- Multinomial naive bayes ---------------
    mnb_final = MultinomialNB()
    mnb_final.fit(X_train, y_train)
    y_predict_mnb = mnb_final.predict(X_test)
    accuracy_mnb.append(accuracy_score(y_test, y_predict_mnb))
    print('-----Multinomial naive bayes ---------- \nTesting accuracy: ', accuracy_mnb)
    print('Confusion matrix \n', confusion_matrix(y_test, y_predict_mnb))
    
    
    # ------------------SVC ---------------------------
    svc_final = SVC(C=1, kernel='linear')
    svc_final.fit(X_train, y_train)
    y_predict_svc = svc_final.predict(X_test)
    accuracy_svc.append(accuracy_score(y_test, y_predict_svc))
    print('-----Support Vector Machines ---------- \nTesting accuracy: ', accuracy_svc)
    print('Confusion matrix \n', confusion_matrix(y_test, y_predict_svc))
    
    # --------- Random Forest --------------------------
    rf_final = RandomForestClassifier(n_estimators=50, max_features=2)
    rf_final.fit(X_train, y_train)
    y_predict_rf = rf_final.predict(X_test)
    accuracy_rfc.append(accuracy_score(y_test, y_predict_rf))
    print('-----Random Forest ---------- \nTesting accuracy: ', accuracy_rfc)
    print('Confusion matrix: \n', confusion_matrix(y_test, y_predict_rf))
    
    # ------- Gradient Boosting Classifier ---------------
    gbc_final = GradientBoostingClassifier(n_estimators=10, learning_rate=0.2, max_features=10)
    gbc_final.fit(X_train, y_train)
    y_predict_gbc = gbc_final.predict(X_test)
    accuracy_gbc.append(accuracy_score(y_test, y_predict_gbc))
    print('-----Gradient Boosting Machines ---------- \nTesting accuracy: ', accuracy_gbc)
    print('Confusion matrix: \n', confusion_matrix(y_test, y_predict_gbc))
  

print('-------------------Using numerical features------------------')
print('Gaussian naive bayes: {0:.3f} (mean) and {1:.3f} (std)'.format(np.mean(accuracy_gnb), np.std(accuracy_gnb)))
print('Multinomial naive bayes: {0:.3f} (mean) and {1:.3f} (std)'.format(np.mean(accuracy_mnb), np.std(accuracy_mnb)))
print('SVC: {0:.3f} (mean) and {1:.3f} (std)'.format(np.mean(accuracy_svc), np.std(accuracy_svc)))
print('RFC: {0:.3f} (mean) and {1:.3f} (std)'.format(np.mean(accuracy_rfc), np.std(accuracy_rfc)))
print('GBC: {0:.3f} (mean) and {1:.3f} (std)'.format(np.mean(accuracy_gbc), np.std(accuracy_gbc)))

#%%Training the random forest model with full dataset and optimized hyperparameters
rf_model = RandomForestClassifier(n_estimators=50, max_features=2)
rf_model.fit(X, y)

# Save the trained model and extracted dataframe
with open ('training_data_model_v1.pickle', 'wb') as handle:
    pickle.dump([rf_model, df_refined], handle, protocol=pickle.HIGHEST_PROTOCOL)

