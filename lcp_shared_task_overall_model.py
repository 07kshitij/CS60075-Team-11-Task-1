# -*- coding: utf-8 -*-
"""
  Overall Architecture used for the LCP shared Task

  Note - Slight deviations are possible if everything is run from scratch due to certain non-determinism in PyTorch computations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy import stats
import spacy
from spacy_syllables import SpacySyllables

from utils import *
from get_features import *
from Models.NeuralNet import NN
from Models.LinearRegression import LinearRegressor
from Models.SVR import SupportVectorRegressor

# Seed all rngs for deterministic results
seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Syllable tagger pipeline
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("syllables", after='tagger')

# Dataset Paths
SINGLE_TRAIN_DATAPATH = "https://raw.githubusercontent.com/MMU-TDMLab/CompLex/master/train/lcp_single_train.tsv"
SINGLE_TEST_DATAPATH = "https://raw.githubusercontent.com/MMU-TDMLab/CompLex/master/test-labels/lcp_single_test.tsv"

MULTI_TRAIN_DATAPATH = "https://raw.githubusercontent.com/MMU-TDMLab/CompLex/master/train/lcp_multi_train.tsv"
MULTI_TEST_DATAPATH = "https://raw.githubusercontent.com/MMU-TDMLab/CompLex/master/test-labels/lcp_multi_test.tsv"

# Obtain the dataFrames
df_train_single, df_test_single, df_train_multi, df_test_multi = get_data_frames(SINGLE_TRAIN_DATAPATH, SINGLE_TEST_DATAPATH, MULTI_TRAIN_DATAPATH, MULTI_TEST_DATAPATH)

single_tokens_train_raw = df_train_single["token"].astype(str).to_list()
single_tokens_test_raw = df_test_single["token"].astype(str).to_list()

y_single_train = df_train_single["complexity"].astype(np.float32).to_numpy()
y_single_test = df_test_single["complexity"].astype(np.float32).to_numpy()

multi_tokens_train_raw = df_train_multi["token"].astype(str).to_list()
multi_tokens_test_raw = df_test_multi["token"].astype(str).to_list()

y_multi_train = df_train_multi["complexity"].astype(np.float32).to_numpy()
y_multi_test = df_test_multi["complexity"].astype(np.float32).to_numpy()

sent_train_single_raw = df_train_single["sentence"].to_list()
sent_test_single_raw = df_test_single["sentence"].to_list()

sent_train_multi_raw = df_train_multi["sentence"].to_list()
sent_test_multi_raw = df_test_multi["sentence"].to_list()

EMBEDDING_DIM = 50

def get_embeddings(EMBEDDING_DIM):
  embedding_index = {}
  with open('glove.6B.{}d.txt'.format(EMBEDDING_DIM), 'r', encoding='utf-8') as f:
    for line in f:
      values = line.split()
      token = values[0]
      embedding_index[token] = np.asarray(values[1:], dtype='float32')
  return embedding_index

embedding_index = get_embeddings(EMBEDDING_DIM)
print('\n[Token Count] GloVE embeddings: {}'.format(len(embedding_index)))

word_to_ix, word_to_ix_multi = map_token_to_idx(sent_train_single_raw, sent_train_multi_raw)
print('\n[Vocab size] Single Word: {}\n[Vocab size] Multi Word: {}'.format(len(word_to_ix), len(word_to_ix_multi)))

""" biLSTM to predict target probability

    Reference - [PyTorch](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
"""

HIDDEN_DIM = 10

"""biLSTM class to calculate token probability given context"""

class biLSTM(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
    super(biLSTM, self).__init__()
    self.hidden_dim = hidden_dim
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
    self.hidden2tag = nn.Linear(2 * hidden_dim, output_size)

  def prepare_embedding(self, sentence):
    embeddings = []
    for word in sentence:
      word = word.lower()
      if word in embedding_index:
        embeddings.extend(embedding_index[word])
      else:
        embeddings.extend(np.random.random(EMBEDDING_DIM).tolist())
    embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
    return embeddings

  def forward(self, sentence):
    sentence = sentence.split()
    embeds = self.prepare_embedding(sentence)
    lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
    tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
    tag_scores = F.softmax(tag_space, dim=1)
    return tag_scores

"""biLSTM model for single word targets"""

model = biLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(word_to_ix))
# Used while training phase to avoid re training the same model again and again
# To use the saved model, change the False in next cell to True
path_biLSTM_single = './TrainedModels/biLSTM.pt'

USE_PRETRAINED_SINGLE_WORD_TARGET_MODEL = True

if USE_PRETRAINED_SINGLE_WORD_TARGET_MODEL:
  print('Using pre-trained biLSTM on single target expressions')
  model = torch.load(path_biLSTM_single)
  model.eval()
else:
  print('Training biLSTM on single target expressions')
  # Train the model for 10 epochs
  model = biLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(word_to_ix))
  loss_function = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  for epoch in range(10):
    loss_sum = 0
    for sentence in sent_train_single_raw:
      model.zero_grad()
      targets = prepare_sequence(sentence, word_to_ix)
      tag_scores = model(sentence)
      loss = loss_function(tag_scores, targets)
      loss_sum += loss
      loss.backward()
      optimizer.step()
    print('Epoch: {} Loss: {}'.format(epoch, loss_sum.item()))

"""biLSTM model for multi word targets"""

model_multi = biLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix_multi), len(word_to_ix_multi))
# Used while training phase to avoid re training the same model again and again
# To use the saved model, change the False in next cell to True
path_biLSTM_multi = './TrainedModels/biLSTM_multi.pt'

USE_PRETRAINED_MULTI_WORD_TARGET_MODEL = True

if USE_PRETRAINED_MULTI_WORD_TARGET_MODEL:
  print('Using pre-trained biLSTM on multi target expressions')
  model_multi = torch.load(path_biLSTM_multi)
  model_multi.eval()
else:
  print('Training biLSTM on multi target expressions')
  model_multi = biLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix_multi), len(word_to_ix_multi))
  loss_function = nn.MSELoss()
  optimizer = optim.Adam(model_multi.parameters(), lr=0.01)
  for epoch in range(10):
    loss_sum = 0
    for sentence in sent_train_multi_raw:
      model_multi.zero_grad()
      targets = prepare_sequence(sentence, word_to_ix_multi)
      tag_scores = model_multi(sentence)
      loss = loss_function(tag_scores, targets)
      loss_sum += loss
      loss.backward()
      optimizer.step()
    print('Epoch: {} Loss: {}'.format(epoch, loss_sum.item()))

print('\n')
print('+++ Generating Train features for Single word expressions +++')
features_train_single = prepare_features_single_word(single_tokens_train_raw, sent_train_single_raw, nlp, word_to_ix, model, embedding_index, EMBEDDING_DIM)
print('+++ [COMPLETE] Feature generation for Train Single word expressions +++')
print('\n')
print('+++ Generating Test features for Single word expressions +++')
features_test_single = prepare_features_single_word(single_tokens_test_raw, sent_test_single_raw, nlp, word_to_ix, model, embedding_index, EMBEDDING_DIM)
print('+++ [COMPLETE] Feature generation for Test Single word expressions +++')
print('\n')

print('+++ Generating Train features for Multi word expressions +++')
features_train_multi = prepare_features_multi_word(multi_tokens_train_raw, sent_train_multi_raw, nlp, word_to_ix_multi, model_multi, embedding_index, EMBEDDING_DIM)
print('+++ [COMPLETE] Feature generation for Train Multi word expressions +++')
print('\n')

print('+++ Generating Test features for Multi word expressions +++')
features_test_multi = prepare_features_multi_word(multi_tokens_test_raw, sent_test_multi_raw, nlp, word_to_ix_multi, model_multi, embedding_index, EMBEDDING_DIM)
print('+++ [COMPLETE] Feature generation for Test Multi word expressions +++')
print('\n')

# Convert all features to torch.tensor to enable use in PyTorch models
X_train_single_tensor = torch.tensor(features_train_single, dtype=torch.float32, device=device)
X_test_single_tensor = torch.tensor(features_test_single, dtype=torch.float32, device=device)
X_train_multi_tensor = torch.tensor(features_train_multi, dtype=torch.float32, device=device)
X_test_multi_tensor = torch.tensor(features_test_multi, dtype=torch.float32, device=device)

# Reshape all output complexity scores to single dimension vectors
y_single_train = y_single_train.reshape(y_single_train.shape[0], -1)
y_single_test = y_single_test.reshape(y_single_test.shape[0], -1)
y_multi_train = y_multi_train.reshape(y_multi_train.shape[0], -1)
y_multi_test = y_multi_test.reshape(y_multi_test.shape[0], -1)

# Convert all target outputs to torch.tensor to enable use in PyTorch models
Y_train_single_tensor = torch.tensor(y_single_train, dtype=torch.float32, device=device)
Y_test_single_tensor = torch.tensor(y_single_test, dtype=torch.float32, device=device)
Y_train_multi_tensor = torch.tensor(y_multi_train, dtype=torch.float32, device=device)
Y_test_multi_tensor = torch.tensor(y_multi_test, dtype=torch.float32, device=device)

# Ensure each sample from test and train for single word expression is taken
output_shape(X_train_single_tensor, X_test_single_tensor, Y_train_single_tensor, Y_test_single_tensor, 0, 0)

# Ensure each sample from test and train for multi word expression is taken
output_shape(X_train_multi_tensor, X_test_multi_tensor, Y_train_multi_tensor, Y_test_multi_tensor, 1, 0)

NUM_EPOCHS = 30

loss_function = nn.MSELoss()

embedding_dim = X_train_single_tensor.shape[1]
model_NN = NN(embedding_dim)
model_NN.to(device)

# Used while training phase to save the best checkpoint (with the best Pearson R on test set)
# To use the saved model, change the False in next cell to True
path_NN = './TrainedModels/NN_0.731.pt'

USE_PRETRAINED_SINGLE_WORD_TARGET_NN = True

if USE_PRETRAINED_SINGLE_WORD_TARGET_NN:
  print('\n +++ Using pre-trained NN on single target expressions +++')
  model_NN = torch.load(path_NN)
  model_NN.eval()
else:
  print('\n +++ Training NN on single target expressions... +++\n')
  model_NN = NN(embedding_dim)
  model_NN.to(device)
  loss_function = nn.MSELoss()
  optimizer = optim.Adam(model_NN.parameters(), lr=0.002)
  for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    out = model_NN(X_train_single_tensor)
    loss = loss_function(out, Y_train_single_tensor)
    loss.backward()
    optimizer.step()
    out_test = model_NN(X_test_single_tensor)
    testR = compute_pearsonR(out_test, Y_test_single_tensor, device)
    trainR = compute_pearsonR(out, Y_train_single_tensor, device)
    print("Epoch {} : Train R = {} | Test R = {}".format(epoch + 1, round(trainR, 6), round(testR, 6)))

out_NN = model_NN(X_test_single_tensor)
print('\n +++ Metrics for Single Word Expression using NN +++ \n')
evaluate_metrics(out_NN, Y_test_single_tensor, device)

embedding_dim = X_train_multi_tensor.shape[1]
model_NN_multi = NN(embedding_dim)
model_NN_multi.to(device)

# Used while training phase to save the best checkpoint (with the best Pearson R on test set)
# To use the saved model, change the False in next cell to True
path_NN_multi = './TrainedModels/NN_multi_0.775.pt'

USE_PRETRAINED_MULTI_WORD_TARGET_NN = True

if USE_PRETRAINED_MULTI_WORD_TARGET_NN:
  print('\n +++ Using pre-trained NN on multi target expressions +++')
  model_NN_multi = torch.load(path_NN_multi)
  model_NN_multi.eval()
else:
  print('\n +++ Training NN on multi target expressions... +++\n')
  model_NN_multi = NN(embedding_dim)
  model_NN_multi.to(device)
  loss_function = nn.MSELoss()
  optimizer = optim.Adam(model_NN_multi.parameters(), lr=0.002)
  for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    out = model_NN_multi(X_train_multi_tensor)
    loss = loss_function(out, Y_train_multi_tensor)
    loss.backward()
    optimizer.step()
    out_test = model_NN_multi(X_test_multi_tensor)
    testR = compute_pearsonR(out_test, Y_test_multi_tensor, device)
    trainR = compute_pearsonR(out, Y_train_multi_tensor, device)
    print("Epoch {} : Train R = {} | Test R = {}".format(epoch + 1, round(trainR, 6), round(testR, 6)))

out_NN_multi = model_NN_multi(X_test_multi_tensor)
print('\n +++ Metrics for Multi Word Expression using NN +++ \n')
evaluate_metrics(out_NN_multi, Y_test_multi_tensor, device)

""" Machine Learning Methods """

X_train_single_np = np.array(features_train_single)
X_test_single_np = np.array(features_test_single)
Y_train_single_np = np.array(y_single_train.reshape(y_single_train.shape[0], -1))
Y_test_single_np = np.array(y_single_test.reshape(y_single_test.shape[0], -1))

print('\n')
output_shape(X_train_single_np, X_test_single_np, Y_train_single_np, Y_test_single_np, 0, 1)

X_train_multi_np = np.array(features_train_multi)
X_test_multi_np = np.array(features_test_multi)
Y_train_multi_np = np.array(y_multi_train.reshape(y_multi_train.shape[0], -1))
Y_test_multi_np = np.array(y_multi_test.reshape(y_multi_test.shape[0], -1))

output_shape(X_train_multi_np, X_test_multi_np, Y_train_multi_np, Y_test_multi_np, 1, 1)

""" Linear Regression """

LR = LinearRegressor()

print('\n +++ Metrics for Single Word Expression using Linear Regression +++ \n')
out_LR = LR.forward(X_train_single_np, Y_train_single_np, X_test_single_np, Y_test_single_np)
evaluate_metrics(out_LR, Y_test_single_np, device)

print('\n +++ Metrics for Multi Word Expression using Linear Regression +++ \n')
out_LR_multi = LR.forward(X_train_multi_np, Y_train_multi_np, X_test_multi_np, Y_test_multi_np)
evaluate_metrics(out_LR_multi, Y_test_multi_np, device)

""" Support Vector Regressor """

svr = SupportVectorRegressor()

print('\n +++ Metrics for Single Word Expression using SVR +++ \n')
out_svr = svr.forward(X_train_single_np, Y_train_single_np, X_test_single_np, Y_test_single_np)
evaluate_metrics(out_svr, Y_test_single_np, device)

print('\n +++ Metrics for Multi Word Expression using SVR +++ \n')
out_svr_multi = svr.forward(X_train_multi_np, Y_train_multi_np, X_test_multi_np, Y_test_multi_np)
evaluate_metrics(out_svr_multi, Y_test_multi_np, device)

single_ids = df_test_single["id"].astype(str).to_list()
multi_ids = df_test_multi["id"].astype(str).to_list()

"""Aggregation of results obtained"""

out_ensemble = []

for idx in range(len(out_NN)):
  score = 0
  score += float(out_NN[idx])
  score += float(out_LR[idx])
  score += float(out_svr[idx])
  score /= 3
  out_ensemble.append(score)
out_ensemble = np.array(out_ensemble)
out_ensemble = out_ensemble.reshape((out_ensemble.shape[0], 1))

print('\n +++ Metrics for Single Word Expression using Overall Model +++ \n')
evaluate_metrics(out_ensemble, Y_test_single_np, device)

out_ensemble_multi = []

for idx in range(len(out_NN_multi)):
  score = 0
  score += float(out_NN_multi[idx])
  score += float(out_LR_multi[idx])
  score += float(out_svr_multi[idx])
  score /= 3
  out_ensemble_multi.append(score)
out_ensemble_multi = np.array(out_ensemble_multi)
out_ensemble_multi = out_ensemble_multi.reshape((out_ensemble_multi.shape[0], 1))

print('\n +++ Metrics for Multi Word Expression using Overall Model +++ \n')
evaluate_metrics(out_ensemble_multi, Y_test_multi_np, device)

output_results(single_ids, multi_ids, out_ensemble, out_ensemble_multi)