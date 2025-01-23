import pandas as pd
import string
import nltk
import numpy as np
import sklearn
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import RobustScaler

NRC_FILE = "corpora/NRC-emotion.txt"
PROFANITY_FILE = "corpora/Profanity_Wordlist.txt"
#credit for this list goes to better_profanity package

class FeatureGenerator:
    """
    A class to generate features for tweets. 
    
    The class contains static methods that operate on pandas dataframes. 
    """
    
    def __init__(self, df):
        self.df = df
        self.threshold = 0.5
        self.feature_names = []

    def get_features(self)-> pd.DataFrame:
        self.features = self.df[self.feature_names]
        return self.features

    def preprocess(self):
        """
        Preprocess data by tokenizing, lowercasing, and removing stopwords. 
        """
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')        
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

        stop_words = set(stopwords.words('english'))
        stop_words.add("user")

        text = self.df['text'].str.lower()
        
        def remove_punctuation(tweet):
            return ''.join(char for char in tweet if char not in string.punctuation)

        text = text.apply(remove_punctuation)

        text = text.apply(word_tokenize)
        
        def filter_stopwords(sentence):
            return [w for w in sentence if not w.lower() in stop_words]

        text = text.apply(filter_stopwords)
        self.df['clean_text'] = text

        lemmatizer = WordNetLemmatizer()
        def lemmatize_tokens(tokens):
            return [lemmatizer.lemmatize(token) for token in tokens]
        
        self.df['lemma_text'] = text.apply(lemmatize_tokens)
        
    def add_punctuation_count(self):
        """
        Counts the number of punctuation marks in each tweet.
        """
        punctuation = string.punctuation

        def count_tweet(tweet): 
            punct_count =  Counter(char for char in tweet if char in punctuation)
            return {p: punct_count.get(p, 0) for p in punctuation}

        text = self.df['text']
        
        punctuation_counts = text.apply(count_tweet)
        
        X_counts = pd.DataFrame(punctuation_counts.tolist(), columns=list(punctuation))

        self.df = pd.merge(self.df, X_counts, left_index=True, right_index=True)
        self.feature_names.extend(X_counts.columns)

    def add_capital_ratio(self):
        """
        Generates new column with the ratio of capital to non-capital letters.
        """
        #remove punctuation to prevent skew
        def remove_punctuation(tweet):
            return ''.join(char for char in tweet if char not in string.punctuation)

        text = self.df['text'].apply(remove_punctuation)
        
        #remove @USER to prevent skew
        def remove_user(tweet: str):
            return tweet.replace("USER", "")
        
        text = text.apply(remove_user)

        #helper function to calculate capital ratio
        def cap_count(tweet: str) -> float:
            total_chars = len(tweet)
            upper_chars = sum(1 for char in tweet if char.isupper())
            return upper_chars / total_chars if total_chars > 0 else 0

        self.df['capitals'] = text.apply(cap_count)
        self.feature_names.append('capitals')

    def add_sentiment_analysis(self):
        """
        Generates sentiment scores using nltk sid module. 
        """
        try:
            nltk.data.find('corpora/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')

        sid = SentimentIntensityAnalyzer()

        def get_sentiment_scores(text):
            return sid.polarity_scores(text)
        
        self.df['sentiment_scores'] = self.df['text'].apply(get_sentiment_scores)

        # Create separate columns for each sentiment score
        self.df['negative'] = self.df['sentiment_scores'].apply(lambda x: x['neg'])
        self.df['neutral'] = self.df['sentiment_scores'].apply(lambda x: x['neu'])
        self.df['positive'] = self.df['sentiment_scores'].apply(lambda x: x['pos'])
        self.df['compound'] = self.df['sentiment_scores'].apply(lambda x: x['compound'])

        # Drop original sentiment score column
        self.df = self.df.drop('sentiment_scores', axis=1)
        self.feature_names.extend(['negative', 'neutral', 'positive', 'compound'])

    def add_str_length(self):
        """
        Generates length column that describes length of original "text" column. 
        """
        
        self.df['length'] = self.df['text'].apply(len)
        self.feature_names.append('length')
    
    def add_count_profanity(self):
        """
        Counts the number of profane words in each tweet.
        """
        profane_words = pd.read_csv(PROFANITY_FILE, header =None, names = ['word'])
        profane_words = profane_words['word'].tolist()

        def count_profane(tweet: list) -> int:
            return sum(1 for word in tweet if word in profane_words)
        
        self.df['profanity'] = self.df['clean_text'].apply(count_profane)
        self.feature_names.append('profanity')

    def scale_features(self):
        """
        Scales the features in a df using robust scaler from scikit-learn.
        """
        #scale features to improve model performance
        scaler = RobustScaler()
        features = self.get_features()
        self.features = pd.DataFrame(scaler.fit_transform(features))
    
    def add_word_count(self, word: str):
        """
        Counts the number of occurences of a given words in each tweet
        
        Accepts string to look for as input. 
        """
        def count_profane(tweet: list) -> int:
            return sum(1 for token in tweet if word == token)
        
        self.df[f"{word}_count"] = self.df['clean_text'].apply(count_profane)
        self.feature_names.append(f"{word}_count")
    
    def add_emotion_count(self, emotion: str):
        """Counts the occurences of words related to a given emotion

        Args:
            emotion (str): Str flag for emotion to search for. Can be 'anger', 'anticipation', 'disgust', 'fear', 'negative', 'positive', 'sadness', 'surprise', 'trust'
        """
        df_nrc = pd.read_csv(NRC_FILE, delimiter='\t', names = ['word', 'emote', 'class'])
        
        #take cases where the word represents the emotion
        emotion_words = df_nrc[(df_nrc['emote'] == emotion) & (df_nrc['class'] == 1)]['word'].tolist()

        def count_profane(tweet: list) -> int:
            return sum(1 for word in tweet if word in emotion_words)
        
        #count occurances
        self.df[emotion] = self.df['lemma_text'].apply(count_profane)
        self.feature_names.append(emotion)
