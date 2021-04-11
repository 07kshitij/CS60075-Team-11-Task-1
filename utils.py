import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy import stats
import random
import os
import csv

# https://www.kaggle.com/bminixhofer/deterministic-neural-networks-using-pytorch
# Seed all rngs for deterministic results

def seed_all(seed = 0):
  random.seed(0)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

# Return dataFrames stored in the URLs in the args

def get_data_frames(SINGLE_TRAIN_DATAPATH, SINGLE_TEST_DATAPATH, MULTI_TRAIN_DATAPATH, MULTI_TEST_DATAPATH):
  df_train_single = pd.read_csv(SINGLE_TRAIN_DATAPATH, sep='\t', quotechar="'", quoting=csv.QUOTE_NONE)
  df_test_single = pd.read_csv(SINGLE_TEST_DATAPATH, sep='\t', quotechar="'", quoting=csv.QUOTE_NONE)

  df_train_multi = pd.read_csv(MULTI_TRAIN_DATAPATH, sep='\t', quotechar="'", quoting=csv.QUOTE_NONE)
  df_test_multi = pd.read_csv(MULTI_TEST_DATAPATH, sep='\t', quotechar="'", quoting=csv.QUOTE_NONE)

  return df_train_single, df_test_single, df_train_multi, df_test_multi

def prepare_sequence(seq, to_ix):
  seq = seq.split()
  idxs = [to_ix[w.lower()] if w.lower() in to_ix else len(to_ix) for w in seq]
  idxs = torch.tensor(idxs)
  idxs = nn.functional.one_hot(idxs, num_classes=len(to_ix))
  idxs = torch.tensor(idxs, dtype=torch.float32)
  return idxs

def map_token_to_idx(sent_train_single_raw, sent_train_multi_raw):
  word_to_ix = {}
  word_to_ix_multi = {}
  for sent in sent_train_single_raw:
    sent = sent.split()
    for word in sent:
      word = word.lower()
      if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix)

  for sent in sent_train_multi_raw:
    sent = sent.split()
    for word in sent:
      word = word.lower()
      if word not in word_to_ix_multi:
        word_to_ix_multi[word] = len(word_to_ix_multi)
  
  return word_to_ix, word_to_ix_multi


# Convert a PyTorch tensor to Numpy ndarray

def convert_tensor_to_np(y, device):
  if device == torch.device("cuda"):
    y = y.cpu()
  y = y.detach().numpy()
  return y

def compute_pearsonR(labels, predicted, device):
  vx, vy = [], []
  if torch.is_tensor(labels):
    vx = labels.clone()
    vx = convert_tensor_to_np(vx, device)
  else:
    vx = deepcopy(labels)
  if torch.is_tensor(predicted):
    vy = predicted.clone()
    vy = convert_tensor_to_np(vy, device)
  else:
    vy = deepcopy(predicted)

  pearsonR = np.corrcoef(vx.T, vy.T)[0, 1]
  return pearsonR

# Evaluate the metrics upon which the model would be evaluated

def evaluate_metrics(labels, predicted, device):
  vx, vy = [], []
  if torch.is_tensor(labels):
    vx = labels.clone()
    vx = convert_tensor_to_np(vx, device)
  else:
    vx = deepcopy(labels)
  if torch.is_tensor(predicted):
    vy = predicted.clone()
    vy = convert_tensor_to_np(vy, device)
  else:
    vy = deepcopy(predicted)

  pearsonR = np.corrcoef(vx.T, vy.T)[0, 1]
  spearmanRho = stats.spearmanr(vx, vy)
  MSE = np.mean((vx - vy) ** 2)
  MAE = np.mean(np.absolute(vx - vy))
  RSquared = pearsonR ** 2

  print(" Peason's R: {}".format(pearsonR))
  print(" Spearman's rho: {}".format(spearmanRho))
  print(" R Squared: {}".format(RSquared))
  print(" Mean Squared Error: {}".format(MSE))
  print(" Mean Average Error: {}".format(MAE))

# Output shape of given matrices. For debugging purposes

def output_shape(X_train, X_test, Y_train, Y_test, target_type, compute_type):
  res_type, data_type = 'Single', 'PyTorch'
  if target_type:
    res_type = 'Multi'
  if compute_type:
    data_type = 'Numpy'
  print('[{} Word]: [{}] X_train shape: {}'.format(res_type, data_type, X_train.shape))
  print('[{} Word]: [{}] X_test shape: {}'.format(res_type, data_type, X_test.shape))
  print('[{} Word]: [{}] Y_train shape: {}'.format(res_type, data_type, Y_train.shape))
  print('[{} Word]: [{}] Y_test shape: {}'.format(res_type, data_type, Y_test.shape))
  print('\n')

# Write output complexity scores to output files

def output_results(single_ids, multi_ids, out_single, out_multi):
  path_out_single = './out_single.csv'
  path_out_multi = './out_multi.csv'

  with open(path_out_single, 'w') as f:
    writer = csv.writer(f)
    for idx in range(len(out_single)):
      writer.writerow([single_ids[idx], float(out_single[idx])])

  with open(path_out_multi, 'w') as f:
    writer = csv.writer(f)
    for idx in range(len(out_multi)):
      writer.writerow([multi_ids[idx], float(out_multi[idx])])

  print('\n +++ Output written to files: {} & {}'.format(path_out_single, path_out_multi))