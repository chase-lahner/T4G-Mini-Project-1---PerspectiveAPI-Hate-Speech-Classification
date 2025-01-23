import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
import FeatureGenerator
import HateSpeechClassifier
import Comparison
import ScoreClassifierClass
from HateSpeechClassifier import *
from FeatureGenerator import *
from MetricsGenerator import *
from ScoreClassifierClass import *
from Comparison import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, f1_score, confusion_matrix

TRAIN_FILE = "civility_data/train.tsv"
DEMOGRAPHIC_FILE = "civility_data/mini_demographic_dev.tsv"
DEV_FILE = "civility_data/dev.tsv"

PERSPECTIVE_DEV_FILE = "civility_data/dev.tsv"
PERSPECTIVE_DEMOGRAPHIC_FILE = "civility_data/mini_demographic_dev.tsv"

def import_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Imports data from string literals.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Returns train, dev, and demographic data. 
    """
    #import data
    df_train = pd.read_csv(TRAIN_FILE, delimiter= "\t")
    df_dev = pd.read_csv(DEV_FILE, delimiter = "\t")
    df_demographic = pd.read_csv(DEMOGRAPHIC_FILE, delimiter= "\t")
    
    #add target column for demographic file 
    df_demographic['label'] = "NOT"

    #return data 
    return df_train, df_dev, df_demographic

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features from tweets

    Args:
        df (pd.Series): Pandas dataframe that contains a column called "text".

    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    # Create the feature generator and preprcess the data
    feature_generator = FeatureGenerator(df)
    feature_generator.preprocess()
    
    # TODO Select the features for your model    
    # To add a feature "uncomment" it by deleting the hashtag in front of the line. 
    # Hover your cursor over the code to find out what it does.
    # You can remove a feature by "commenting" it out, or putting a hashtag in front of the line. 
    
    feature_generator.add_capital_ratio()
    #feature_generator.add_punctuation_count()
    #feature_generator.add_str_length()
    #feature_generator.add_count_profanity()
    #feature_generator.add_sentiment_analysis()
    
    # TODO create custom features for your model   
    # You can customize the two features below for specific words or emotions. Add as many as you would like!   
    # You can reuse a line of code by copying and pasting, just make sure to change the word.
    feature_generator.add_emotion_count("joy")  # for this you can choose the emotions:  'anger', 'anticipation', 'disgust', 'fear', 'negative', 'positive', 'sadness', 'surprise', 'trust'
    feature_generator.add_word_count("hate")    # for this you can choose any word you want

    
    # Scale features to improve model performance and return data
    feature_generator.scale_features()
    return feature_generator.get_features()

def generate_classifier(X, y)-> tuple[LogisticRegression, HateSpeechClassifier]:
    """Generates model from engineered features and original data. 

    Args:
        X (pd.DataFrame): Engineered features df. 
        y (pd.Series): Series containing true labels for df 

    Returns:
        LogisticRegression: Model trained using train. 
    """

    # create the classifier object
    classifier = HateSpeechClassifier(X, y)
    
    # TODO select how long to search for the best model 
    # Low will be the fastest but high will give you the best model possible (it might take 5+ minutes)
    model = classifier.generate_model('med') #low, med, or high
    
    # TODO Select how accurate you want your model to bed. Enter a percentage as a decimal value (0.7 = 70% accurate)
    # If you pick an accuracy too high, the model to default to whatever the closest value is. 
    classifier.optimize_threshold(goal_accuracy=0.7)
    
    return model, classifier

def generate_predictions(classifier, df_dev, df_demographic):
    #generate features
    X_dev = generate_features(df_dev)
    X_demographics = generate_features(df_demographic)
    
    #generate predictions    
    pred_dev = classifier.predict(X_dev)
    pred_dem = classifier.predict(X_demographics)
    
    return pred_dev, pred_dem

def generate_perspective(df_dev, df_demographic):
    # TODO select a threshold for the perspectiveAPI
    # The threshold determines how sensitive the model is to toxicity. 
    # For example, 0.7 means that tweets that are 70% toxic or more will be classified as hate speech.
    threshold = 0.7
    
    #create the classifier object
    perspective = ScoreClassifierClass(df_dev, df_demographic, threshold)

    df_dev['pred'] = perspective.classify_dev()
    df_demographic['pred'] = perspective.classify_demographic()
    
    metrics_dev_class = perspective.run_metrics_dev()
    metrics_demographic_class = perspective.run_metrics_dem()
    
    fpr_class = perspective.fpr()
    fpr_demo_class = perspective.fpr_demographic()
    
    return fpr_class, fpr_demo_class, metrics_dev_class

def run_metrics(df_dev, df_demographic, pred_dev, pred_dem):
    metrics = MetricsGenerator(df_dev, df_demographic, pred_dev, pred_dem)
    fpr = metrics.fpr()
    metrics_dev = metrics.run_metrics()
    fpr_demo = metrics.fpr_demographic()
    metrics.test_false_positives()
    
    compare = Comparison(fpr, fpr_demo, metrics_dev, fpr_class, fpr_demo_class, metrics_dev_class)

    compare.compare()
    
    return fpr, fpr_demo, metrics_dev
    
if __name__ == "__main__":
    #load data
    df_train, df_dev, df_demographic = import_data()
    
    #perspectiveAPI model
    fpr_class, fpr_demo_class, metrics_dev_class = generate_perspective(df_dev, df_demographic)
    
    #create custom model
    X_train = generate_features(df_train)
    model, classifier = generate_classifier(X_train, df_train['label'])    
    df_dev['pred'], df_demographic['pred'] = generate_predictions(classifier, df_dev, df_demographic)

    #run evaluations
    fpr, fpr_demo, metrics_dev = run_metrics(df_dev, df_demographic, df_dev['pred'], df_demographic['pred'])