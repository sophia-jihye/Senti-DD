from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import os

DIRECTIONAL_WORDS_FILEPATH = r'C:\Users\Jihye Park\OneDrive\Ph.D\연구\02.Financial Sentiment Lexicon\Data\Directional_words.csv'

stemmer = PorterStemmer()
lemmatizer=WordNetLemmatizer()
directional_words_df = pd.read_csv(DIRECTIONAL_WORDS_FILEPATH)

def create_polar_sentences(df):
    df = df[df.label != 'neutral']
    return df

def assign_direction_dependency_type(text, label):
    tokens = word_tokenize(text)
    up_cnt, down_cnt = 0, 0
    for token in tokens:
        if stemmer.stem(token) in directional_words_df[directional_words_df['label']=='up'].stemmed.values:
            up_cnt += 1
        if stemmer.stem(token) in directional_words_df[directional_words_df['label']=='down'].stemmed.values:
            down_cnt += 1
    score = up_cnt - down_cnt
    if (score > 0 and label == 'positive') or (score < 0 and label == 'negative'): return 'proportional'
    if (score > 0 and label == 'negative') or (score < 0 and label == 'positive'): return 'inversely_proportional'

def get_preprocessed_nouns(text):
    words = word_tokenize(text)
    nouns = [token for token, tag in pos_tag(words) if tag in ['NN', 'NNS', 'NNP', 'NNPS']]
    nouns = [lemmatizer.lemmatize(token) for token in nouns if len(token)>1]
    return np.array(nouns)

def select_frequent_tokens(list_of_tokens, min_count):
    vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False, min_df=min_count)
    vectorizer.fit_transform(list_of_tokens)
    selected_tokens = vectorizer.vocabulary_.keys()
    selected_tokens = [item for item in selected_tokens if stemmer.stem(item) not in directional_words_df.stemmed.values]
    return selected_tokens

def count_sentences_containing(list_of_tokens, word):
    count = 0
    for tokens in list_of_tokens:
        if word in tokens:
            count += 1
    return count

def count_sentences_not_containing(list_of_tokens, word):
    count = 0
    for tokens in list_of_tokens:
        if word not in tokens:
            count += 1
    return count

def pmi(df, word, t):
    n = len(df)
    a = count_sentences_containing(df[df.direction_dependency==t].nouns, word)
    b = count_sentences_containing(df[df.direction_dependency!=t].nouns, word)
    c = count_sentences_not_containing(df[df.direction_dependency==t].nouns, word)
    return (n*a)/((a+b)*(a+c))

def pmi_combined(df, word):
    pmi_prop = pmi(df, word, 'proportional')
    pmi_invprop = pmi(df, word, 'inversely_proportional')
    if pmi_prop > pmi_invprop: return abs(pmi_prop)
    elif pmi_prop < pmi_invprop: return -abs(pmi_invprop)
    else: return 0
    
def extract_token_score(tokens, scores, t):
    if t == 'proportional':
        if np.max(scores) <= 0: return (None, None)
        return (tokens[np.argmax(scores)], np.max(scores))
    elif t == 'inversely_proportional':
        if np.max(scores) >= 0: return (None, None)
        return (tokens[np.argmin(scores)], np.min(scores))
    
def construct_senti_dd(train_filepath, dd_filepath, senti_dd_filepath):
    df = pd.read_csv(train_filepath)
    log_content = 'Number of headlines = {}'.format(len(df))
    
    # Direction-dependency Type Tagging for Each Sentence
    df = create_polar_sentences(df)    
    log_content += '\nNumber of polar headlines = {}'.format(len(df))
    
    df['direction_dependency'] = df.apply(lambda x: assign_direction_dependency_type(x['headline'], x['label']), axis=1)
    df.dropna(subset=['direction_dependency'], inplace=True)

    df['nouns'] = df['headline'].apply(lambda x: get_preprocessed_nouns(x))
    selected_tokens = select_frequent_tokens(df.nouns.values, 6)
    log_content += '\n\nSelected tokens ({})\n{}'.format(len(selected_tokens), selected_tokens)
    
    df['nouns'] = df['nouns'].apply(lambda nouns: [token for token in nouns if token in selected_tokens])
    df = df[df['nouns'].map(len) != 0]
    log_content += '\n\nNumber of proportional type headlines = {}\t inversely_proportional type headlines = {}'.format(len(df[df.direction_dependency=='proportional']), len(df[df.direction_dependency=='inversely_proportional']))
    
    # Estimation of the Strength of Association Between a Word and a Direction-DependencyType
    lexicon_df = pd.DataFrame({'token': selected_tokens})
    lexicon_df['pmi'] = lexicon_df['token'].apply(lambda x: pmi_combined(df, x))
    
    # Extraction of Direction-dependent Words
    df['scores'] = df['nouns'].apply(lambda x: np.array([lexicon_df[lexicon_df.token==token].iloc[0].pmi for token in x]))

    df['token_score'] = df.apply(lambda x: extract_token_score(x['nouns'], x['scores'], x['direction_dependency']), axis=1)
    df['entity'] = df['token_score'].apply(lambda x: x[0])
    df['entity_score'] = df['token_score'].apply(lambda x: x[1])
    df.drop(columns=['token_score'], inplace=True)

    dd = pd.DataFrame.from_records(list(zip(df.direction_dependency.values, df.entity.values, df.entity_score.values)), columns=['direction_dependency', 'entity', 'score'])
    dd.dropna(subset=['score'], inplace=True)
    dd.drop_duplicates(subset=['direction_dependency', 'entity'], inplace=True)

    proportional_words = dd[dd.direction_dependency=='proportional'].entity.values
    inversely_proportional_words = dd[dd.direction_dependency=='inversely_proportional'].entity.values
    log_content += '\n\nProportional type entities ({})\n{}'.format(len(proportional_words), ', '.join(proportional_words))
    log_content += '\nInversely proportional type entities ({})\n{}'.format(len(inversely_proportional_words), ', '.join(inversely_proportional_words))

    # Post-processing to filter out noisy data
    # all of the characters should be alphabet & number of characters > 2
    dd['survive_post_processing'] = dd['entity'].apply(lambda x: x.isalpha() and len(x)>2)
    dd = dd[dd['survive_post_processing']==True]
    dd.drop(columns=['survive_post_processing'], inplace=True)
    dd.to_csv(dd_filepath, index=False)
    print('Created', dd_filepath)
    
    # Senti-DD Construction based on the List of Directional Words and the Direction-dependent Words
    up_words = directional_words_df[directional_words_df.label=='up'].stemmed.values
    down_words = directional_words_df[directional_words_df.label=='down'].stemmed.values

    records = []
    records.extend([('positive', entity, direction) for entity in proportional_words for direction in up_words])
    records.extend([('positive', entity, direction) for entity in inversely_proportional_words for direction in down_words])
    records.extend([('negative', entity, direction) for entity in proportional_words for direction in down_words])
    records.extend([('negative', entity, direction) for entity in inversely_proportional_words for direction in up_words])
    senti_dd = pd.DataFrame.from_records(records, columns=['sentiment', 'entity', 'directional_word'])
    log_content += '\n\nNumber of positive pairs: {}\t negative pairs: {}'.format(len(senti_dd[senti_dd.sentiment=='positive']), len(senti_dd[senti_dd.sentiment=='negative']))

    senti_dd.to_csv(senti_dd_filepath, index=False)
    print('Created', senti_dd_filepath)
    
    with open(os.path.join(os.path.dirname(train_filepath), "senti_dd_construction.log"), "w") as f:
        f.write(log_content)