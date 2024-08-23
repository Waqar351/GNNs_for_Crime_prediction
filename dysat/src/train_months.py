import torch
import pickle
import pathlib
import argparse
import sys
import os
import time

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the dataloader directory to the Python path
dataloader_dir = os.path.join(current_dir, 'dataloader')
sys.path.append(dataloader_dir)

# Add the directory containing the module to the path
from dataloader.minibatch import MyDataset, get_time_period, select_time_steps
from torch.utils.data import DataLoader
import utils
from models.Dysat_model import DySAT
# from utils import to_device
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import json
import logging


# utils.set_random_seeds()
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

parser = argparse.ArgumentParser()
parser.add_argument('--time_delta', type=str, required=True, default='M', choices=["D", "M", "Y"],
                    help="Time period to consider when grouping crimes in (Y)ear, (D)ay or (M)onth. Default is (M)onth.")

# Parse known arguments first to get the time_delta value
args, unknown= parser.parse_known_args()

ts = select_time_steps(args.time_delta)

parser.add_argument('--time_steps', type=int, nargs='?', default= ts,            # Time steps (Sample Size)
                    help="total time steps used for train, eval and test")

# Experimental settings.

parser.add_argument('--epochs', type=int, nargs='?', default=5000,
                    help='# epochs')
parser.add_argument('--batch_size', type=int, nargs='?', default=32,  #512 default
                    help='Batch size (# nodes)')
parser.add_argument("--early_stop", type=int, default=60,
                    help="patient")

# Tunable hyper-params
parser.add_argument('--residual', type=bool, nargs='?', default=True,
                    help='Use residual')
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,     # 0.01 in paper
                    help='Initial learning rate for self-attention model.')
parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.4,       ### 0.4 has better result
                    help='Spatial (structural) attention Dropout (1 - keep probability).')
parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.4,      ### 0.4 has better result
                    help='Temporal attention Dropout (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,  #0.0005
                    help='Initial learning rate for self-attention model.')
parser.add_argument('--threshold', type=float, nargs='?', default=None,                  # Threshold
                    help='Class threshold used at the output of last layer.')
parser.add_argument('--class_weight', type=float, nargs='?', default=4.420,                  # 4.420 class_weight
                    help='Class weight for minority class.')

# Architecture params
parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
                    help='Encoder layer config: # attention heads in each GAT layer')
parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128',
                    help='Encoder layer config: # units in each GAT layer')
parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                    help='Encoder layer config: # attention heads in each Temporal layer')
parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
                    help='Encoder layer config: # units in each Temporal layer')
parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                    help='Position wise feedforward')
parser.add_argument('--window', type=int, nargs='?', default=-1,
                    help='Window for temporal attention (default : -1 => full)')
args = parser.parse_args()


td = get_time_period(args.time_delta)
reading_path = pathlib.Path(f"./data/dataloader/dysat/label_column_static_feats/{td}") 
# reading_path = pathlib.Path(f"E:/Dysat_model/data/dataloader/dysat/label_column_static_feats/{td}")

# Directory to save model and logs
save_dir = './Results/static_features/model_checkpoints' 
os.makedirs(save_dir, exist_ok=True)

# Function to save model state and metrics
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

# Function to save logs
def save_logs(logs, filename="logs.json"):
    # Convert logs to native Python objects
    logs_native = utils.convert_numpy_to_native(logs)
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(logs_native, f, indent=4)


print('Loading data...')
with open(reading_path / f"dataset_static__{args.time_delta}.pickle", "rb") as f:
    dataset = pickle.load(f)

# ## Splitting data into train, validation and test
seq_data = utils.extract_consecutive_samples(dataset[0]['graphs'], args.time_steps)
tr_seq, val_seq, tes_seq = utils.split_data(seq_data, train_ratio = 0.8, valid_ratio = 0.25, seed=42)

## Create datasets
train_dataset = utils.GraphSequenceDataset(tr_seq)
val_dataset = utils.GraphSequenceDataset(val_seq)
test_dataset = utils.GraphSequenceDataset(tes_seq)

## Creating batches
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn= utils.custom_collate, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, collate_fn= utils.custom_collate, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, collate_fn= utils.custom_collate, pin_memory=True)

print('Dataloader Created!')

## Model initialization
class_weight = torch.Tensor([args.class_weight])
model = DySAT(args, dataset[0]['graphs'][0].x.shape[1], args.time_steps, class_weight).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)



logs = {'train_loss': [], 'val_metrics': [], 'test_metrics': [],'epcoh_duration': []}
logging.basicConfig(filename=os.path.join(save_dir, 'training.log'), level=logging.INFO)

best_epoch_val = 0
best_epoch_bal_val = 0
patient = 0

print('\n Training Starts...')

for epoch in range(args.epochs):
    start_time = time.time()  # Start time of the epoch
    epoch_loss, valid_loss, train_pred, gradient_norm = utils.model_train(epoch, model, train_loader, val_loader, opt, device)
    avg_epoch_loss = np.nanmean(epoch_loss)
    avg_valid_loss = np.nanmean(valid_loss)
    avg_grad_norm = np.mean(gradient_norm)
    
    torch.cuda.empty_cache()
    print('Training Done!')

    print('Validation Start!')
    prc_val_pos, rec_val_pos, f1_val_pos, auc_val_pos, mcc_val, balanced_acc_val, prc_val_neg, rec_val_neg, f1_val_neg, auc_val_neg, thr_val, misclassif_rate_val, tp_val, fp_val, fn_val, tn_val = utils.model_eval(model, val_loader, args.threshold, device, bvalid= True)
    print('Validation Done!')

    prc_test_pos, rec_test_pos, f1_test_pos, auc_test_pos, mcc_test, balanced_acc_test, prc_test_neg, rec_test_neg, f1_test_neg, auc_test_neg, thr_test, misclassif_rate_test, tp_test, fp_test, fn_test, tn_test = utils.model_eval(model, test_loader, thr_val, device, bvalid= False)
    print('Testing Done!\n')

    ## Logging loss and evaluation metrics for each epoch
    logging.info(f"Epoch {epoch+1}, Avg_loss: {avg_epoch_loss}, Avg_Validation_loss: {avg_valid_loss}, Avg_gradient_norm: {avg_grad_norm}")
    logging.info(f"Epoch {epoch+1}, Precision_val_pos: {prc_val_pos}, Recall_val_pos: {rec_val_pos}, F1_val_pos: {f1_val_pos}, auc_val_pos: {auc_val_pos},  mcc_val = {mcc_val}, balanced_acc_val = {balanced_acc_val}, Precision_val_neg: {prc_val_neg}, Recall_val_neg: {rec_val_neg}, F1_val_neg: {f1_val_neg}, auc_val_neg: {auc_val_neg}, threshold_val: {thr_val}, misClassification_rate_val: {misclassif_rate_val}, True_pos_val: {tp_val}, False_pos_val: {fp_val}, False_neg_val: {fn_val}, True_neg_val: {tn_val} ")
    logging.info(f"Epoch {epoch+1}, Precision_test_pos: {prc_test_pos}, Recall_test_pos: {rec_test_pos}, F1_test_pos: {f1_test_pos}, auc_test_pos: {auc_test_pos},  mcc_test = {mcc_test}, balanced_acc_test = {balanced_acc_test}, Precision_test_neg: {prc_test_neg}, Recall_test_neg: {rec_test_neg}, F1_test_neg: {f1_test_neg}, auc_test_neg: {auc_test_neg}, threshold_test: {thr_test}, misClassification_rate_test: {misclassif_rate_test}, True_pos_test: {tp_test}, False_pos_test: {fp_test}, False_neg_test: {fn_test}, True_neg_test: {tn_test}")

    # Log metrics manual
    logs['train_loss'].append({'epoch' : epoch + 1, 'Avg_loss': avg_epoch_loss,'Avg_Validation_loss': avg_valid_loss ,'Avg_gradient_norm': avg_grad_norm})
    logs['val_metrics'].append({'epoch': epoch + 1, 'Precision_val_pos': prc_val_pos, 'Recall_val_pos': rec_val_pos, 'F1_val_pos': f1_val_pos, 'auc_val_pos': auc_val_pos, 'mcc_val': mcc_val, 'balanced_acc_val': balanced_acc_val, 'Precision_val_neg': prc_val_neg, 'Recall_val_neg': rec_val_neg, 'F1_val_neg': f1_val_neg, 'auc_val_neg': auc_val_neg, 'threshold_val': thr_val, 'misClassification_rate_val': misclassif_rate_val, 'True_pos_val': tp_val, 'False_pos_val': fp_val, 'False_neg_val': fn_val, 'True_neg_val': tn_val})
    logs['test_metrics'].append({'epoch': epoch + 1, 'Precision_test_pos': prc_test_pos, 'Recall_test_pos': rec_test_pos, 'F1_test_pos': f1_test_pos, 'auc_test_pos': auc_test_pos, 'mcc_test': mcc_test, 'balanced_acc_test': balanced_acc_test, 'Precision_test_neg': prc_test_neg, 'Recall_test_neg': rec_test_neg, 'F1_test_neg': f1_test_neg, 'auc_test_neg': auc_test_neg, 'threshold_test': thr_test, 'misClassification_rate_test': misclassif_rate_test, 'True_pos_test': tp_test, 'False_pos_test': fp_test, 'False_neg_test': fn_test, 'True_neg_test': tn_test})
     
    
    print("Epoch {:<3},  Avg_loss = {:.3f}, Avg_Validation_loss = {:.3f}, Avg_gradient_norm = {:.3f}".format(epoch + 1, avg_epoch_loss, avg_valid_loss, avg_grad_norm))
    print("Epoch {:<3},  Precision_val_pos = {:.3f}, Recall_val_pos = {:.3f}, F1_val_pos = {:.3f}, auc_val_pos = {:.3f}, mcc_val = {:.3f}, balanced_acc_val = {:.3f}".format(epoch + 1, prc_val_pos, rec_val_pos, f1_val_pos, auc_val_pos, mcc_val, balanced_acc_val))
    print("Epoch {:<3},  Precision_test_pos = {:.3f}, Recall_test_pos = {:.3f}, F1_test_pos = {:.3f}, auc_test_pos = {:.3f}, mcc_test = {:.3f}, balanced_acc_test = {:.3f}".format(epoch + 1, prc_test_pos, rec_test_pos, f1_test_pos, auc_test_pos, mcc_test, balanced_acc_test))
    print("Epoch {:<3},  Precision_val_neg = {:.3f}, Recall_val_neg = {:.3f}, F1_val_neg = {:.3f}, auc_val_neg = {:.3f}, threshold_val= {:.3f}".format(epoch + 1, prc_val_neg, rec_val_neg, f1_val_neg, auc_val_neg, thr_val))
    print("Epoch {:<3},  Precision_test_neg = {:.3f}, Recall_test_neg = {:.3f}, F1_test_neg = {:.3f}, auc_test_neg = {:.3f}, threshold_test = {:.3f}".format(epoch + 1, prc_test_neg, rec_test_neg, f1_test_neg, auc_test_neg, thr_test))


    # Saving best model at each epoch
    if auc_val_pos > best_epoch_val or balanced_acc_val > best_epoch_bal_val:
        best_epoch_val = auc_val_pos
        best_epoch_bal_val = balanced_acc_val
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(),
            'avg_epoch_loss': avg_epoch_loss,
            'best_auc': best_epoch_val,
            'best_bal_auc': best_epoch_bal_val,
            'test_auc': auc_test_pos,
            'threshold': thr_test,
        }, filename=os.path.join(save_dir, 'model_best.pth.tar'))
        logging.info(f"Saved model at epoch {epoch+1} with train loss {avg_epoch_loss}")
        patient = 0
    else:
        patient += 1
        if patient > args.early_stop:
            print('Early stop')
            logging.info(f"Early stop at patient {patient}")
            break
    
    end_time = time.time()  # End time of the epoch
    epoch_duration = (end_time - start_time) / 60  # Duration of the epoch in minutes

    ## Logging epcoh duration
    logging.info(f"Epoch {epoch+1}, Duration: {epoch_duration:.2f} minutes, Patient: {patient}")
    logs['epcoh_duration'].append({'epoch': epoch + 1, 'Duration': epoch_duration, 'Patient' : patient})

    # Save manual logs
    save_logs(logs, filename=os.path.join(save_dir, 'training_logs_manual.json'))

print('Epochs Complete!')

# Path to the checkpoint file
checkpoint_path = f"{save_dir}/model_best.pth.tar"

# Loading the best model
checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint['state_dict'])
opt.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
best_auc = checkpoint['best_auc']
thr_upd = checkpoint['threshold']

model.eval()

print(f"Loaded checkpoint from epoch {epoch} with best AUC: {best_auc}")

prc_test_pos, rec_test_pos, f1_test_pos, auc_test_pos, mcc_test, balanced_acc_test, prc_test_neg, rec_test_neg, f1_test_neg, auc_test_neg, thr_test, misclassif_rate_test, tp_test, fp_test, fn_test, tn_test = utils.model_eval(model, test_loader, thr_upd, device, bvalid= False)

print('Precision:', prc_test_pos)
print('Recall:', rec_test_pos)
print('F1:', f1_test_pos)
print('AUC:', auc_test_pos)
print('Balanced AUC:', balanced_acc_test)
print('MissClassificationRate:', misclassif_rate_test)