import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, matthews_corrcoef, balanced_accuracy_score
import matplotlib.pyplot as plt
import pathlib
import pickle
import sys
from sklearn.model_selection import train_test_split
from dataloader.minibatch import MyDataset, get_time_period
# # Add the directory containing the module to the path
# sys.path.append('E:/Dysat_model/dataloader/')
# import minibatch
# from minibatch import MyDataset
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--time_delta', type=str, required=True, default='M', choices=["D", "M", "Y"],
                    help="Time period to consider when grouping crimes in (Y)ear, (D)ay or (M)onth. Default is (M)onth.")
parser.add_argument('--model', type=str, required=True, default='logistic', choices=["logistic", "randomforest", "svm", "decisiontrees"],
                    help="Which traditional machine learning models to use for classification.")

args = parser.parse_args()

td = get_time_period(args.time_delta)


reading_path = pathlib.Path(f"E:/Dysat_model/data/dataloader/dysat/label_column_static_feats/{td}") # original
# reading_path = pathlib.Path(f"E:/Dysat_model/data/dataloader/dysat/feature_analysis/{td}")
# reading_path = pathlib.Path(f"E:/Dysat_model/data/dataloader/dysat/feature_analysis/num_crimes_Months_totCrime_highway_2_26_dynamic_stat_feats/{td}")  # All 55 features

with open(reading_path / f"features_all_times__{args.time_delta}.pickle", "rb") as f:
    features_all_times = pickle.load(f)
# with open(reading_path / f"features_analysis__{args.time_delta}.pickle", "rb") as f:
#     features_analysis = pickle.load(f)


def preprocess_features(features):   # row-wise normalization or row-wise scaling
        """Row-normalize feature matrix and convert to tuple representation"""
        matrix = np.array(features.todense())     # Convert the feature into dense matrix of [#nodes, Features]
        features = matrix[:, :-1]               #Excluding last column
        labels = matrix[:, -1]
        rowsum = np.array(features.sum(1))          # In this particular case the sum is 1 for each row
        r_inv = np.power(rowsum, -1).flatten()      # Take the power of 1 raise to -1 and then flatten it to have single dimenion
        r_inv[np.isinf(r_inv)] = 0.                 # Find infinite indices and replace it to 0.
        r_mat_inv = sp.diags(r_inv)                 # Create a sparse diagonal matrix with the inverse row sums
        features = r_mat_inv.dot(features)          # Row-normalize the feature matrix ##each row of the feature matrix is element-wise multiplied by the corresponding reciprocal of the row sum
        
        return features, labels

def extract_features_target(sparse_matrix):
    features, label = preprocess_features(sparse_matrix)
    features = np.array(features)
    label = np.array(label)
    # Convert the numpy array to a pandas DataFrame
    features = pd.DataFrame(features)
    label = pd.DataFrame(label)
    return features, label

def temporal_feature_generator(dt):
    features_df = pd.DataFrame()
    for i in range(len(dt)):
        sample = dt[i]
        dynamic_feature = pd.DataFrame()
        for j in range(len(sample) - 1):
            static_feats = pd.DataFrame(np.array(sample[0].todense())[:, :-2])
            target_feats = pd.DataFrame(np.array(sample[-1].todense())[:, -1] )

            # dynamic_feature[f'crime_t{j}'] = np.array(sample[j].todense())[:, -2]
        
        # df_concat = pd.concat([static_feats, dynamic_feature, target_feats], axis=1)
        df_concat = pd.concat([static_feats, target_feats], axis=1)
        features_df = pd.concat([features_df,df_concat], axis = 0)
        # breakpoint()
        # Convert DataFrame to Sparse Matrix
        sparse_matrix = sp.csr_matrix(features_df.values)
    return sparse_matrix


# Initialize lists to store evaluation metrics
## precisions, recalls, f1s, aucs, accuracies = [], [], [], [], []

seq_data = utils.extract_consecutive_samples(features_all_times, 3)
train_seq, test_seq =  train_test_split(seq_data, test_size = 0.2, random_state=42, shuffle = False)

features_all_times[0]
matrix = np.array(features_all_times[0].todense())
        

train_data = temporal_feature_generator(train_seq)
test_data = temporal_feature_generator(test_seq)

# Extract features and target for training and testing
X_train, y_train = extract_features_target(train_data)
X_test, y_test = extract_features_target(test_data)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# breakpoint()

# Train the model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
if args.model == 'logistic':
    clf = LogisticRegression(class_weight= 'balanced')
if args.model == 'randomforest':
    clf = RandomForestClassifier(class_weight= 'balanced')
if args.model == 'svm':
    clf = SVC(class_weight= 'balanced', random_state=42)
if args.model == 'decisiontrees':
    clf = DecisionTreeClassifier(class_weight= 'balanced', random_state=42)

    

clf.fit(X_train, np.ravel(y_train))
# Make predictions
predic_prob = clf.predict_proba(X_test)[:, 1]

pd.DataFrame(predic_prob).to_csv(f'{args.model}_output_probs_Dyanmic_features.csv', index=False)

predictions = [1 if pred >= 0.5 else 0 for pred in predic_prob]
    
# Metrics for negative class
precision = precision_score(y_test, predictions, pos_label= 1)
recall = recall_score(y_test, predictions, pos_label= 1)
f1 = f1_score(y_test, predictions, pos_label= 1)
auc = roc_auc_score(y_test, predic_prob)
accuracy = accuracy_score(y_test, predictions)

mcc = matthews_corrcoef(y_test, predictions)  # Matthews Correlation Coefficient
balanced = balanced_accuracy_score(y_test, predictions)       # Balanced Accuracy
misclassification_rate = np.mean(np.array(y_test) != np.array(predictions))

# Metrics for negative class
precision_neg = precision_score(y_test, predictions, pos_label= 0)
recall_neg = recall_score(y_test, predictions, pos_label= 0)
f1_neg = f1_score(y_test, predictions, pos_label= 0)
auc_neg = roc_auc_score(y_test, predictions)
# accuracy_neg = accuracy_score(y_test, -predic_prob)


# Print average metrics
print(f'Precision Positive: {precision :.4f}')
print(f'Recall Positive: {recall:.4f}')
print(f'F1 Score Positive: {f1:.4f}')
print(f'AUC Positive: {auc:.4f}')
print(f'Accuracy Positive: {accuracy:.4f}')

print(f'Mathew Coefficient: {mcc:.4f}')
print(f'Balanced Accuracy: {balanced:.4f}')
print(f'MisClassification Rate: {misclassification_rate:.4f}')

# Print average metrics
print(f'Precision Negative: {precision_neg :.4f}')
print(f'Recall Negative: {recall_neg:.4f}')
print(f'F1 Score Negative: {f1_neg:.4f}')
print(f'AUC Negative: {auc_neg:.4f}')
# print(f'Accuracy Negative: {accuracy_neg:.4f}')