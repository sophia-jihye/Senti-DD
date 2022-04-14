from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm 
tqdm.pandas()
import pandas as pd
import os, torch, copy, shutil
import numpy as np

import gensim.downloader as api
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
import reports

root_dir = './data' 
# train_filepaths = sorted(glob(os.path.join(root_dir, 'FinancialPhrase*', '*', 'train.csv')))   # FPB
# train_filepaths = sorted(glob(os.path.join(root_dir, 'SemEval*', '*', 'train.csv'))) # SemEval
train_filepaths = sorted(glob(os.path.join(root_dir, 'FiQA*', '*', 'train.csv'))) # FiQA

model_names = ['word2vec-google-news-300']

def do_prepare_data(relabel_dict, filepath):
    df = pd.read_csv(filepath)[['headline', 'label']]
    df.columns = ['text', 'label']
    print('Loaded {}'.format(filepath))
    df['label'] = df['label'].apply(lambda x: relabel_dict[x])
    return df
    
def get_X(model, sentences):
    def convert_text_into_vector(model, text):
        words = word_tokenize(text)
        vectors = []
        for word in words:
            try: vectors.append(model[word])
            except: continue
        return np.vstack(vectors).mean(axis=0)
    return np.vstack([convert_text_into_vector(model, sentence) for sentence in sentences])

def start_test(clf, X_test, df, save_dir, postfix=''):
    print('Inferencing..\n')
    df['predicted_label'] = clf.predict(X_test)
    
    # Save results
    df['correct'] = df.apply(lambda x: x.true_label==x.predicted_label, axis=1)
    labels, preds = df.true_label, df.predicted_label
    accuracy = len(df[df['correct']==True]) / len(df)

    csv_filepath = os.path.join(save_dir, 'predictions_{}.csv'.format(postfix))
    df.to_csv(csv_filepath, index=False)
    
    report_filepath = os.path.join(save_dir, 'classification_report_{}.csv'.format(postfix))
    reports.create_classification_report(labels, preds, accuracy, report_filepath)
    
    confusion_filepath = os.path.join(save_dir, 'confusion_matrix_{}.csv'.format(postfix))
    reports.create_confusion_matrix(labels, preds, confusion_filepath)

if __name__ == '__main__':        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relabel_dict = {'negative':0, 'neutral':1, 'positive':2}
    num_classes = len(relabel_dict)    

    for train_filepath in train_filepaths:
        test_filepath = train_filepath.replace('train.csv', 'test.csv')
        
        train_df = do_prepare_data(relabel_dict, train_filepath)
        train_texts, y_train = train_df['text'].values, train_df['label'].values
        
        for model_name in model_names:
            save_dir = os.path.dirname(train_filepath).replace('data', 'results')
            if not os.path.exists(save_dir): os.makedirs(save_dir)
                
            # word2vec features
            model = api.load(model_name)
            X_train = get_X(model, train_texts)
            
            # logistic regression
            clf = LogisticRegression(random_state=0)
            clf.fit(X_train, y_train)

            ##### test #####
            test_df = do_prepare_data(relabel_dict, test_filepath)
            test_df.rename(columns = {'label' : 'true_label'}, inplace = True)
            
            X_test = get_X(model, test_df['text'].values)
            start_test(clf, X_test, test_df, save_dir, postfix=model_name)