import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import pathlib
import pickle
import sys
# Add the directory containing the module to the path
sys.path.append('E:/Dysat_model/dataloader/')
import minibatch
from minibatch import MyDataset
import numpy as np
import scipy.sparse as sp
from scipy.sparse import vstack


td = 'Months'
reading_path = pathlib.Path(f"E:/Dysat_model/data/dataloader/dysat/label_column_static_feats/{td}")
t_delta = 'M'

with open(reading_path / f"features_all_times__{t_delta}.pickle", "rb") as f:
    features_all_times = pickle.load(f)


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


# Initialize lists to store evaluation metrics
precisions, recalls, f1s, aucs, accuracies = [], [], [], [], []

# Sliding window approach for training and testing
for cnt, i in enumerate(range(len(features_all_times) - 3)):
    # Train on two months and predict the next month
    train_data_1 = features_all_times[i]
    train_data_2 = features_all_times[i + 1]
    test_data = features_all_times[i + 2]
    
    # Combine the training data
    train_data = vstack((train_data_1, train_data_2))
    
    # Extract features and target for training and testing
    X_train, y_train = extract_features_target(train_data)
    X_test, y_test = extract_features_target(test_data)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    
    # Train the model
    #model = RandomForestClassifier(n_estimators=100, random_state=42)
    model = LogisticRegression(class_weight= 'balanced')
    model.fit(X_train, np.ravel(y_train))
    
    # Make predictions
    predic_prob = model.predict_proba(X_test)[:, 1]
    # pd.DataFrame(predic_prob).to_csv('logistic_regression_output_probs.csv', index=False)
    pd.DataFrame(predic_prob).to_csv('Random_forest_output_probs.csv', index=False)
    predictions = [1 if pred >= 0.5 else 0 for pred in predic_prob]
    
    # Calculate metrics
    precision = precision_score(y_test, predictions, pos_label= 0)
    recall = recall_score(y_test, predictions, pos_label= 0)
    f1 = f1_score(y_test, predictions, pos_label= 0)
    auc = roc_auc_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    aucs.append(auc)
    accuracies.append(accuracy)

print(cnt)
# Print average metrics
print(f'Average Precision: {sum(precisions) / len(precisions):.4f}')
print(f'Average Recall: {sum(recalls) / len(recalls):.4f}')
print(f'Average F1 Score: {sum(f1s) / len(f1s):.4f}')
print(f'Average AUC: {sum(aucs) / len(aucs):.4f}')
print(f'Average Accuracy: {sum(accuracies) / len(accuracies):.4f}')