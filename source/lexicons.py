import pandas as pd

import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download('sentiwordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag

from textblob import TextBlob

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

LM_FILEPATH = '/media/dmlab/My Passport/DATA/Lexicon/LM_Word_List/LM_Word_List.csv'

vader_analyser = SentimentIntensityAnalyzer()
stemmer = PorterStemmer()
lemmatizer=WordNetLemmatizer()
lm_df = pd.read_csv(LM_FILEPATH)

def lm_score(text):
    tokens = word_tokenize(text)
    pos_cnt, neg_cnt = 0, 0
    for token in tokens:
        if token in lm_df[lm_df['label']=='positive'].word.values:
            pos_cnt += 1
        if token in lm_df[lm_df['label']=='negative'].word.values:
            neg_cnt += 1
    score = pos_cnt - neg_cnt
    return score

def senti_dd_polarity(text, senti_dd):
    def senti_dd_score(text, senti_dd):
        tokens = word_tokenize(text)
        pos_cnt, neg_cnt = 0, 0
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        for _, row in senti_dd.iterrows():
            if row.entity in lemmatized_tokens and row.directional_word in stemmed_tokens:
                if row.sentiment=='positive': pos_cnt += 1
                elif row.sentiment=='negative': neg_cnt += 1
        score = pos_cnt - neg_cnt
        return score
    
    score = lm_score(text)
    context_sentiment_score = senti_dd_score(text, senti_dd)
    if context_sentiment_score > 0: score += 1
    elif context_sentiment_score < 0: score -= 1
    
    if score > 0: return 'positive'
    elif score < 0: return 'negative'
    else: return 'neutral'
    
def lm_polarity(text):
    score = lm_score(text)
    if score > 0: return 'positive'
    elif score < 0: return 'negative'
    else: return 'neutral'
    
def vader_polarity(text):
    score = vader_analyser.polarity_scores(text)['compound']
    if score >= 0.05: return 'positive'
    elif score <= -0.05: return 'negative'
    else: return 'neutral'

def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
    def penn_to_wn(tag):
        """
        Convert between the PennTreebank tags to simple Wordnet tags
        """
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        return None
    
    sentiment = 0.0
    tokens_count = 0
    
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))

        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())

            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1

    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 'neutral'

    # sum greater than 0 => positive sentiment
    if sentiment > 0: return 'positive'
    elif sentiment < 0: return 'negative'
    else: return 'neutral'    
    
def textblob_polarity(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0: return 'positive'
    elif score < 0: return 'negative'
    else: return 'neutral'    