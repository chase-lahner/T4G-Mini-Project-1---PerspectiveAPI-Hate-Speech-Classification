import sklearn
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

class ScoreClassifierClass:
    def __init__(self, df_dev, df_demographic, threshold):
        self.df_dev = df_dev
        self.df_demographic = df_demographic
        self.threshold = threshold
        self.col_names = ['label', 'pred']

    def classify_dev(self):
        """
        Classifies the data based on the threshold.
        """
        cur_df = self.df_dev

        if 'perspective_score' not in cur_df.columns:
            raise ValueError("The DataFrame must contain a 'perspective_score' column.")
        
        X = self.df_dev['perspective_score']
        y = X > self.threshold

        return y.replace({True: 'OFF', False: 'NOT'})

    
    def classify_demographic(self):
        """
        Classifies the data based on the threshold.
        """
        cur_df = self.df_demographic

        if 'perspective_score' not in cur_df.columns:
            raise ValueError("The DataFrame must contain a 'perspective_score' column.")
        
        X = self.df_demographic['perspective_score']
        y = X > self.threshold

        return y.replace({True: 'OFF', False: 'NOT'})
    
    def run_metrics_dev(self):
        """
        Runs metrics to evaluate results for classification
        """

        cur_df = self.df_dev

        y_true = cur_df[self.col_names[0]]
        y_pred = cur_df[self.col_names[1]]

        accuracy = accuracy_score(y_true, y_pred)
        labels = ['OFF', 'NOT']
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0, labels = labels)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
        }
    
    def run_metrics_dem(self):
        """
        Runs metrics to evaluate results for classification
        """

        cur_df = self.df_demographic

        y_true = cur_df[self.col_names[0]]
        y_pred = cur_df[self.col_names[1]]

        accuracy = accuracy_score(y_true, y_pred)
        labels = ['OFF', 'NOT']
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0, labels = labels)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
        }
    
    def fpr(self):
        """
        Calculates fpr for a df that contains the actual and prediction values.
        """

        cur_df = self.df_dev

        y_true = cur_df[self.col_names[0]]
        y_pred = cur_df[self.col_names[1]]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return fpr
    
    def fpr_arg(self, df):

        y_true = df['label']
        y_pred = df['pred']

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['NOT', 'OFF']).ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return fpr
    
    def fpr_demographic(self):
        """
        Generate FPR for each unique demographic in a dataframe
        """

        cur_df = self.df_demographic

        unique_demographics = cur_df['demographic'].unique()

        fpr_by_demographic = {}

        for demographic in unique_demographics:
            fpr_by_demographic[demographic] = self.fpr_arg(cur_df[cur_df['demographic'] == demographic])

        return fpr_by_demographic
    
    def test_false_positives(self):
        y_true = self.df_dev['label']
        y_pred = self.df_dev['pred']

        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.max_colwidth', None)  # Show full content of each cell   

        false_positives = (y_pred == 'OFF') & (y_true == 'NOT')

        print("------------------------------------------------------")

        print("Number of False Positives PERSPECTIVE API: ", sum(false_positives))

        print("------------------------------------------------------")

        print("\n")

        false_positives_index = false_positives[false_positives].index

        sampled_false_positives = false_positives_index.to_series().sample(n=min(10, len(false_positives_index)))

        print("Sampled False Positives: ")
        for entry in sampled_false_positives:
            print(self.df_dev.loc[entry,'text'])
            print("\n")

        print("------------------------------------------------------")
        print("\n")