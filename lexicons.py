import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('sentiwordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from textblob import TextBlob
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from afinn import Afinn
from sentistrength import PySentiStr

LM_FILEPATH = '/media/dmlab/My Passport/DATA/Lexicon/LM_Word_List/LM_Word_List.csv'
senti_strength_jar_filepath = '/media/dmlab/My Passport/DATA/Lexicon/SentiStrength/SentiStrengthCom.jar'
senti_strength_data_dirname = '/media/dmlab/My Passport/DATA/Lexicon/SentiStrength/SentiStrengthDataEnglishOctober2019/'
mpqa_filepath = '/media/dmlab/My Passport/DATA/Lexicon/MPQA_Subjectivity/subjclueslen1-HLTEMNLP05.csv'
socal_filepath = '/media/dmlab/My Passport/DATA/Lexicon/SO-CAL/adj_adv_noun_verb.csv'
sentiment140_filepath = '/media/dmlab/My Passport/DATA/Lexicon/Sentiment140/unigrams-pmilexicon.csv'

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
    
vader_analyser = SentimentIntensityAnalyzer()
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
    
afinn = Afinn()
def afinn_polarity(text):
    score = afinn.score(text)
    if score > 0: return 'positive'
    elif score < 0: return 'negative'
    else: return 'neutral' 
    
senti = PySentiStr()
senti.setSentiStrengthPath(senti_strength_jar_filepath) 
senti.setSentiStrengthLanguageFolderPath(senti_strength_data_dirname) 
def sentistrength_polarity(text):
    score = senti.getSentiment([text])[0]
    if score > 0: return 'positive'
    elif score < 0: return 'negative'
    else: return 'neutral' 

mpqa_df = pd.read_csv(mpqa_filepath)
def mpqa_polarity(text):
    tokens = word_tokenize(text)
    pos_cnt, neg_cnt = 0, 0
    for token in tokens:
        if token in mpqa_df[mpqa_df['priorpolarity']=='positive'].word.values:
            pos_cnt += 1
        if token in mpqa_df[mpqa_df['priorpolarity']=='negative'].word.values:
            neg_cnt += 1
    score = pos_cnt - neg_cnt
    if score > 0: return 'positive'
    elif score < 0: return 'negative'
    else: return 'neutral' 

socal_df = pd.read_csv(socal_filepath)
convert_word_to_socal_score = dict(zip(socal_df['word'], socal_df['polarity_score']))
def socal_polarity(text):
    tokens = word_tokenize(text)
    score = sum([convert_word_to_socal_score.setdefault(word, 0) for word in tokens])
    if score > 0: return 'positive'
    elif score < 0: return 'negative'
    else: return 'neutral' 

sentiment140_df = pd.read_csv(sentiment140_filepath)
convert_word_to_140_score = dict(zip(sentiment140_df['word'], sentiment140_df['polarity_score']))
def sentiment140_polarity(text):
    tokens = word_tokenize(text)
    score = sum([convert_word_to_140_score.setdefault(word, 0) for word in tokens])
    if score >= 0.05: return 'positive'
    elif score <= -0.05: return 'negative'
    else: return 'neutral'