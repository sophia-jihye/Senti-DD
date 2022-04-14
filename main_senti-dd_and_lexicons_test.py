import glob, os
import pandas as pd

from reports import create_confusion_matrix, create_classification_report, average_classification_report, sum_confusion_matrix
import lexicons
from senti_dd_construction import construct_senti_dd

data_dir = './data'
# target_data_dir = os.path.join(data_dir, 'FinancialPhrase*', '*') # FPB
target_data_dir = os.path.join(data_dir, 'SemEval*', '*') # SemEval

train_filepaths = glob.glob(os.path.join(target_data_dir, 'train.csv'))
test_filepaths = glob.glob(os.path.join(target_data_dir, 'test.csv'))

result_dir = './results'
# target_result_dir = os.path.join(result_dir, 'FinancialPhrase*_DS{}*', '*') # FPB
target_result_dir = os.path.join(result_dir, 'SemEval*', '*') # SemEval

report_filepaths =os.path.join(target_result_dir, 'classification_report_{}.csv')
conf_filepaths =os.path.join(target_result_dir, 'confusion_matrix_{}.csv')
predictions_filepaths =os.path.join(target_result_dir, 'predictions_{}.csv')

if __name__ == '__main__':

    # Construct Senti-DD
    for train_filepath in train_filepaths:
        print('Processing {}..'.format(train_filepath))
        dd_filepath = os.path.join(os.path.dirname(train_filepath), 'direction_dependent_entities.csv')
        senti_dd_filepath = os.path.join(os.path.dirname(train_filepath), 'Senti-DD.csv')
        construct_senti_dd(train_filepath, dd_filepath, senti_dd_filepath)

    lexicons = [(lexicons.senti_dd_polarity, 'senti-dd'), (lexicons.lm_polarity, 'lm'), (lexicons.vader_polarity, 'vader'), \
                (lexicons.swn_polarity, 'swn'), (lexicons.textblob_polarity, 'textblob'), (lexicons.sentiment140_polarity, 'Sen140'), \
                (lexicons.socal_polarity, 'SO-CAL'), (lexicons.mpqa_polarity, 'MPQA'), (lexicons.afinn_polarity, 'AFINN'), \
                (lexicons.sentistrength_polarity, 'SentiStrength')]

    Predict polarity based on lexicons
    for test_filepath in test_filepaths:
        for lexicon_func, lexicon_name in lexicons:
            print('Processing {} with {}..'.format(test_filepath, lexicon_name))
            df_filepath = os.path.join(os.path.dirname(test_filepath), 'predictions_{}.csv')
            conf_filepath = os.path.join(os.path.dirname(test_filepath), 'confusion_matrix_{}.csv')
            report_filepath = os.path.join(os.path.dirname(test_filepath), 'classification_report_{}.csv')

            df = pd.read_csv(test_filepath)

            if lexicon_name == 'senti-dd':
                senti_dd_filepath = os.path.join(os.path.dirname(test_filepath), 'Senti-DD.csv')
                senti_dd = pd.read_csv(senti_dd_filepath)
                df['prediction'] = df['headline'].apply(lambda x: senti_dd_polarity(x, senti_dd))
            else:         
                df['prediction'] = df['headline'].apply(lambda x: lexicon_func(x))

            df['correct'] = df.apply(lambda x: x['label']==x['prediction'], axis=1)
            df.to_csv(df_filepath.format(lexicon_name), index=False)

            labels, preds = df.label, df.prediction
            create_confusion_matrix(labels, preds, conf_filepath.format(lexicon_name))
            accuracy = len(df[df['correct']==True]) / len(df)
            create_classification_report(labels, preds, accuracy, report_filepath.format(lexicon_name))