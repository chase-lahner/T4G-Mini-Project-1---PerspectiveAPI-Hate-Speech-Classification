import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
import FeatureGenerator
import HateSpeechClassifier
from HateSpeechClassifier import *
from FeatureGenerator import *
from MetricsGenerator import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, f1_score, confusion_matrix

TRAIN_FILE = "civility_data/civility_data/train.tsv"
DEMOGRAPHIC_FILE = "civility_data/civility_data/mini_demographic_dev.tsv"
DEV_FILE = "civility_data/civility_data/dev.tsv" 

def fpr(df: pd.DataFrame) -> float:
    """Calculates fpr for a df that contains the actual and prediction values.

    Args:
        df (pd.DataFrame): Pandas df that contains 'label' and 'y_pred' column

    Returns:
        float: FPR for the dataframe.
    """
    #extract true and predicted values from the df
    y_true = df['label']
    y_pred = df['y_pred']

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = ['NOT', 'OFF']).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return fpr
    
def fpr_demographic(df: pd.DataFrame, y_pred: pd.Series) -> dict:
    """Generate FPR for each unique demographic in a dataframe

    Args:
        df (pd.DataFrame): Pandas df containing 'demographic' column and 'label' column.
        y_pred (pd.Series): Pandas series with the predicted values for the df.

    Returns:
        dict: Dictionary with the FPR for each demographic.
    """
    #add predictions to df
    df['y_pred'] = y_pred
    
    #extract all demographics
    unique_demographics = df['demographic'].unique()
    
    #initalize dictionary to hold fpr for each demographic
    fpr_by_demographic = {}
    
    #calculate FPR for each demographic
    for demographic in unique_demographics:
        fpr_by_demographic[demographic] = fpr(df[df['demographic'] == demographic])
        
    return fpr_by_demographic

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features from tweets

    Args:
        df (pd.Series): Pandas dataframe that contains a column called "text".

    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
        
    feature_generator = FeatureGenerator(df)
    feature_generator.preprocess()
    feature_generator.add_punctuation_count()
    feature_generator.add_str_length()
    feature_generator.add_capital_ratio()
    feature_generator.add_count_profanity()
    feature_generator.add_sentiment_analysis()
    feature_generator.scale_features()
    return feature_generator.get_features(), feature_generator.feature_names

def explain_all_predictions(model, feature_matrix, feature_names):
    """
    Explains the contributions of all features for all rows in a dataset.

    Args:
        model: Trained logistic regression model.
        feature_matrix: Design matrix (DataFrame or NumPy array) containing feature values for all rows.
        feature_names: List of feature names.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to an input row,
                      and each column corresponds to the contribution of a feature.
    """
    # Get feature contributions
    contributions = feature_matrix * model.coef_[0]  # Element-wise multiplication of all rows
    total_scores = contributions.sum(axis=1) + model.intercept_[0]  # Add intercept to each row
    
    # Combine into a DataFrame
    contributions_df = pd.DataFrame(contributions, columns=feature_names)
    contributions_df['Raw Prediction'] = total_scores
    contributions_df['Final Prediction'] = (total_scores >= 0.5).astype(int)  # Threshold for binary classification (0.5)

    return contributions_df

import pandas as pd
from sklearn.metrics import confusion_matrix

def calculate_group_fpr(df, group_column, true_label_column, pred_label_column):
    """
    Calculates the False Positive Rate (FPR) for each group in a specified demographic column.

    Args:
        df (pd.DataFrame): DataFrame containing demographic, true labels, and predictions.
        group_column (str): Name of the column representing demographic groups.
        true_label_column (str): Name of the column representing true labels.
        pred_label_column (str): Name of the column representing predicted labels.

    Returns:
        pd.DataFrame: A DataFrame with group names and their corresponding FPR.
    """
    fpr_data = []
    
    # Loop through each group
    for group, group_df in df.groupby(group_column):
        y_true = group_df[true_label_column]
        y_pred = group_df[pred_label_column]

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['NOT', 'OFF']).ravel()
        
        # Calculate FPR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_data.append({'Demographic': group, 'FPR': fpr})
    
    return pd.DataFrame(fpr_data)

if __name__ == "__main__":
    # Import data
    df_train = pd.read_csv(TRAIN_FILE, delimiter="\t")
    df_dev = pd.read_csv(DEV_FILE, delimiter="\t")
    df_demographic = pd.read_csv(DEMOGRAPHIC_FILE, delimiter="\t")
    
    # Add target column for demographic file
    df_demographic['label'] = "NOT"
    print("Data imported successfully!")
    
    # Generate features and scales (scaling features improves model performance)
    X_train, feature_names = generate_features(df_train)
    print("Features generated successfully!")
    
    # Train model
    classifier = HateSpeechClassifier(X_train, df_train['label'])
    model = classifier.generate_model()
    classifier.select_threshold()
    print("Model Trained!")
    
    # Generate features for development and demographic datasets
    X_dev, _ = generate_features(df_dev)
    X_demographics, _ = generate_features(df_demographic)
    print("Dev features generated")
    
    # Adjust threshold and make predictions
    df_dev['pred'] = classifier.predict(X_dev)
    df_demographic['pred'] = classifier.predict(X_demographics)
    
    # Get feature weights and visualize
    feature_weights = model.coef_[0]  # Get weights of the logistic regression model

    # Combine feature names and weights into a DataFrame
    weights_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': feature_weights
    }).sort_values(by='Weight', ascending=False)

    # Print and save feature weights
    print("Feature Weights:")
    print(weights_df)
    weights_df.to_csv("feature_weights.csv", index=False)

    # Plot top 10 features by absolute weight
    top_features = weights_df.reindex(weights_df['Weight'].abs().sort_values(ascending=False).index).head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'], top_features['Weight'], color='blue')
    plt.xlabel('Weight')
    plt.ylabel('Feature')
    plt.title('Top 10 Feature Weights')
    plt.gca().invert_yaxis()
    plt.show()

    # Explain contributions for the entire training set
    all_explanations = explain_all_predictions(model, X_train, feature_names)

    # Save to CSV for further analysis
    all_explanations.to_csv("all_feature_contributions.csv", index=False)

    # Print the first few rows to inspect
    print(all_explanations.head())

    # Run evaluations
    metrics = MetricsGenerator(df_dev, df_demographic, df_dev['pred'], df_demographic['pred'])
    fpr = metrics.fpr()
    metrics_dev = metrics.run_metrics()
    fpr_demo = metrics.fpr_demographic()

    print(metrics_dev)
    print(fpr_demo)

    
    # Calculate FPR for each demographic group
    group_fpr = calculate_group_fpr(df_demographic, group_column='demographic', true_label_column='label', pred_label_column='pred')

    # Print FPR results
    print(group_fpr)
    # Plot group-specific FPR
    plt.figure(figsize=(10, 6))
    plt.bar(group_fpr['Demographic'], group_fpr['FPR'], color='salmon')
    plt.ylim(0, 1)  # FPR ranges from 0 to 1
    plt.title('False Positive Rates by Demographic Group', fontsize=16)
    plt.xlabel('Demographic Group', fontsize=14)
    plt.ylabel('False Positive Rate (FPR)', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()