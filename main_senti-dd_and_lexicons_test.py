import os
from glob import glob
import pandas as pd

from reports import create_confusion_matrix, create_classification_report, average_classification_report, sum_confusion_matrix
from senti_dd_construction import construct_senti_dd
import lexicons

data_dir = './data'
# target_data_dir = os.path.join(data_dir, 'FinancialPhrase*', '*') # FPB
# target_data_dir = os.path.join(data_dir, 'SemEval*', '*') # SemEval
target_data_dir = os.path.join(data_dir, 'FiQA*', '*') # FiQA

train_filepaths = sorted(glob(os.path.join(target_data_dir, 'train.csv'))) 
test_filepaths = sorted(glob(os.path.join(target_data_dir, 'test.csv'))) 

if __name__ == '__main__':

    lexicon_list = [(lexicons.senti_dd_polarity, 'senti-dd'), (lexicons.lm_polarity, 'lm'), (lexicons.vader_polarity, 'vader'), \
                (lexicons.swn_polarity, 'swn'), (lexicons.textblob_polarity, 'textblob'), (lexicons.sentiment140_polarity, 'Sen140'), \
                (lexicons.socal_polarity, 'SO-CAL'), (lexicons.mpqa_polarity, 'MPQA'), (lexicons.afinn_polarity, 'AFINN'), \
                (lexicons.sentistrength_polarity, 'SentiStrength')]

    # Construct Senti-DD
    for train_filepath in train_filepaths:
        save_dir = os.path.dirname(train_filepath).replace('data', 'results')
        
        for test_filepath in test_filepaths:
            print('Processing {}..'.format(train_filepath))
            dd_filepath = os.path.join(save_dir, 'direction_dependent_entities.csv')
            senti_dd_filepath = os.path.join(save_dir, 'Senti-DD.csv')
            construct_senti_dd(train_filepath, dd_filepath, senti_dd_filepath)
            for lexicon_func, lexicon_name in lexicon_list:
                print('Processing {} with {}..'.format(test_filepath, lexicon_name))

                df = pd.read_csv(test_filepath)

                if lexicon_name == 'senti-dd':
                    senti_dd_filepath = os.path.join(save_dir, 'Senti-DD.csv')
                    senti_dd = pd.read_csv(senti_dd_filepath)
                    df['prediction'] = df['headline'].apply(lambda x: lexicons.senti_dd_polarity(x, senti_dd))
                else:         
                    df['prediction'] = df['headline'].apply(lambda x: lexicon_func(x))

                df['correct'] = df.apply(lambda x: x['label']==x['prediction'], axis=1)
                df.to_csv(os.path.join(save_dir, 'predictions_{}.csv'.format(lexicon_name), index=False)

                labels, preds = df.label, df.prediction
                create_confusion_matrix(labels, preds, \
                                        os.path.join(save_dir, 'confusion_matrix_{}.csv'.format(lexicon_name))
                accuracy = len(df[df['correct']==True]) / len(df)
                create_classification_report(labels, preds, accuracy, \
                                             os.path.join(save_dir, 'classification_report_{}.csv'.format(lexicon_name)))
