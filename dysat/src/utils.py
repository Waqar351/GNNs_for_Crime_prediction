import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_curve, confusion_matrix
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
from torch_geometric.data import Batch

class GraphSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_graphs = self.sequences[idx]
        return input_graphs

def custom_collate(batch):
    # batch is a list of sequences, each sequence is a list of Data objects (graphs for each time step)
    num_time_steps = len(batch[0])  # Assume all sequences have the same number of time steps
    batched_graphs = [[] for _ in range(num_time_steps)]
    
    for sequence in batch:
        for t, graph in enumerate(sequence):
            batched_graphs[t].append(graph)
    
    # Convert lists of graphs to Batch objects
    batched_graphs = [Batch.from_data_list(time_step_graphs) for time_step_graphs in batched_graphs]
    
    return batched_graphs


def to_device(batch, device):
    feed_dict = copy.deepcopy(batch)
    graphs = feed_dict
    feed_dict = [g.to(device) for g in graphs]

    return feed_dict

def extract_consecutive_samples(data, time_series_size):
    """
    Extracts consecutive time series samples (number of Graphs) from the main list.
    
    Args:
    - data (list): The main list containing data.
    - time_series_size (int): The size of each time series sample (number of graphs in each sample).
    
    Returns:
    - List[List]: A list of samples, where each sample is a list of elements.
    """
    return [data[i:i + time_series_size] for i in range(len(data) - time_series_size + 1)]


def split_data(data, train_ratio, valid_ratio,  seed=42):
    
    # Calculate split indices
    total_len = len(data)
    train_end = int(train_ratio * total_len)
    
    # Split the data
    train_data = data[:train_end]
    test_data = data[train_end:]

    train_data, val_data = train_test_split(train_data, test_size = valid_ratio, random_state=seed, shuffle = True)
    
    return train_data, val_data, test_data

def adjusted_classify_count(y_actual, y_predicted):
    cm = confusion_matrix(y_actual, y_predicted)
    prevalence = cm.sum(axis=1) / cm.sum()
    est_prevalence = cm.sum(axis=0) / cm.sum()
    adjustment = prevalence / est_prevalence
    return adjustment

# def adjust_predictions(y_pred_proba, adjustment):
def adjust_predictions(y_pred_proba, adjustment, adj_ratio):
    adjusted_proba = y_pred_proba * adjustment * adj_ratio
    # return np.argmax(adjusted_proba, axis=1)
    # Normalize to ensure they sum to 1
    # adjusted_proba = adjusted_proba / np.sum(adjusted_proba)
    return adjusted_proba

def model_eval(model, data, threshold, device, bvalid):
    model.eval()
    
    y_true = []
    y_pred = []
    binary_pred_output = []
    # precision, recall, f1, auc, mcc, balanced_acc = 0,0,0,0,0,0

    with torch.no_grad():
        
        for idx, feed_dict in enumerate(data):
            feed_dict = to_device(feed_dict, device)
            pred_output = model(feed_dict)
            act_output = feed_dict[-1].y
            # pd.DataFrame(pred_output.cpu().numpy()).to_csv(f'Pred_evaluation_output_{idx}.csv', index=False)
            # pd.DataFrame(act_output.cpu().numpy()).to_csv(f'actual_evaluation_output_{idx}.csv', index=False)
            y_true.extend(act_output.cpu().numpy())
            y_pred.extend(pred_output.cpu().detach().numpy())
            # print(y_pred)
        
        # pd.DataFrame(y_pred).to_csv('Pred_evaluation_output.csv', index=False)

        #####
        ###### Calculation prior proportion
        # calculate Prior Train
        # Concatenate labels from all graphs except the last one
        concatenated_labels = np.concatenate([graph.y.cpu().numpy() for graph in feed_dict[:-1]])

        # Assuming y_test is a numpy array of the labels for the test set
        unique_classes_tr, counts_tr = np.unique(concatenated_labels, return_counts=True)
        prior_train = counts_tr / len(concatenated_labels)



        # Assuming y_test is a numpy array of the labels for the test set
        unique_classes_te, counts_te = np.unique(act_output.cpu().numpy(), return_counts=True)
        prior_test = counts_te / len(act_output.cpu().numpy())
        adjust_ratio = prior_test/prior_train
        
        #####

        y_pred = np.array(y_pred)

        if bvalid:
            best_threshold = 0
            # Compute Precision-Recall curve
            precs, recs, thrs = precision_recall_curve(y_true, y_pred)
            # Compute F1 score for each threshold
            f1_scores = 2 * (precs * recs) / (precs + recs + 1e-10)  # Avoid division by zero
            # Find the best threshold based on F1 score
            best_threshold_idx = np.argmax(f1_scores)
            best_threshold = thrs[best_threshold_idx]
            best_f1 = f1_scores[best_threshold_idx]

            threshold = best_threshold
        else:
            threshold = threshold

        # binary_pred_output_old = [1 if pred >= threshold else 0 for pred in y_pred]
        binary_pred_output_old = [1 if pred >= 0.5 else 0 for pred in y_pred]

        ### Adjust the Classifier output using Quantification
        # Estimate the class distribution in the test data
        adjustment = adjusted_classify_count(y_true, binary_pred_output_old)

        # breakpoint()
        # Adjust the predictions
        # adjusted_proba = adjust_predictions(y_pred, adjustment[0])
        adjusted_proba = adjust_predictions(y_pred, adjustment[0], adjust_ratio[0])

        # binary_pred_output = [1 if pred >= threshold else 0 for pred in adjusted_proba]
        binary_pred_output = [1 if pred >= threshold else 0 for pred in adjusted_proba]

        # breakpoint()
        # Metrics for positive class
        prec_pos = precision_score(y_true, binary_pred_output, pos_label = 1)
        rec_pos = recall_score(y_true, binary_pred_output, pos_label = 1)
        f1_pos = f1_score(y_true, binary_pred_output, pos_label = 1)
        auc_pos = roc_auc_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, binary_pred_output)  # Matthews Correlation Coefficient
        balanced = balanced_accuracy_score(y_true, binary_pred_output)       # Balanced Accuracy
        misclassification_rate = np.mean(np.array(y_true) != np.array(binary_pred_output))
        cm = confusion_matrix(y_true, binary_pred_output)
        tp, fp, fn, tn = cm.ravel()

        # Metrics for negative class
        prec_neg = precision_score(y_true, binary_pred_output, pos_label = 0)
        rec_neg = recall_score(y_true, binary_pred_output, pos_label = 0)
        f1_neg = f1_score(y_true, binary_pred_output, pos_label = 0)
        auc_neg = roc_auc_score(y_true, -y_pred)



        return prec_pos, rec_pos, f1_pos, auc_pos, mcc, balanced, prec_neg, rec_neg, f1_neg, auc_neg, threshold, misclassification_rate, tp, fp, fn, tn
    

def model_train(epoch, model, train_data,val_data, optimizer, device):
    model.train()

    epoch_loss = []
    train_pred = []
    grad_norm_all = []
    
    for idx, feed_dict in enumerate(train_data):
        feed_dict = to_device(feed_dict, device)
        optimizer.zero_grad()
        loss, pred = model.get_loss(feed_dict)
        loss.backward()
        ###########################################################
        # Log gradient norms
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        logging.info(f"Epoch {epoch+1}, Sample {idx + 1},Gradient_Norm: {grad_norm}")
        grad_norm_all.append(grad_norm)
        ################################################################
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss.append(loss.item())
        train_pred.append(pred)
        logging.info(f"Epoch {epoch+1}, Sample_{idx + 1}_Loss: {loss.item()}")
        print('Epoch:', epoch + 1, 'Sample', idx+1)
    
    # Calculate validation loss
    model.eval()
    val_loss = []
    with torch.no_grad():
        for idx, feed_dict in enumerate(val_data):
            feed_dict = to_device(feed_dict, device)
            loss, _ = model.get_loss(feed_dict)
            val_loss.append(loss.item())
    
    return epoch_loss, val_loss, train_pred, grad_norm_all

# Custom function to convert numpy objects to native Python objects
def convert_numpy_to_native(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(i) for i in obj]
    return obj

def load_checkpoint(filepath, model, optimizer):
    """
    Load the model and optimizer state from a checkpoint file.
    
    Parameters:
    - filepath (str): Path to the checkpoint file.
    - model (torch.nn.Module): The model to load the state dictionary into.
    - optimizer (torch.optim.Optimizer, optional): The optimizer to load the state dictionary into.
    
    Returns:
    - checkpoint (dict): The entire checkpoint dictionary.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint, model

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# Function to compare columns
def compare_columns(df):
    n = len(df.columns)
    duplicate_columns = {}
    zero_columns = []

    for i in range(n):
        col1 = df.columns[i]
        if (df[col1] == 0).all():
            zero_columns.append(col1)
        for j in range(i + 1, n):
            col2 = df.columns[j]
            if df[col1].equals(df[col2]):
                if col1 not in duplicate_columns:
                    duplicate_columns[col1] = []
                duplicate_columns[col1].append(col2)
    
    return duplicate_columns, zero_columns