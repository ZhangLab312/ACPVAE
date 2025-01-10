import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from timm.optim import Lamb
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW,AutoModelForSequenceClassification,AutoModel
import torch
from torch.nn import LSTM, Linear, CrossEntropyLoss
from sklearn.model_selection import train_test_split
from torch import nn


# Creating a dataset class
class SequenceClassificationDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'sequence_text': sequence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('from_pretrained/prot_bert_bfd')

# Creating data loaders for training and test sets
train_dataset = SequenceClassificationDataset(
    sequences=sequences_train,
    labels=labels_train,
    tokenizer=tokenizer,
    max_length=50
)
train_data_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_dataset = SequenceClassificationDataset(
    sequences=sequences_test,
    labels=labels_test,
    tokenizer=tokenizer,
    max_length=50
)
test_data_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

import torch
import torch.nn as nn
from transformers import BertModel


class BertWithCustomFeaturesAndAttention(nn.Module):
    def __init__(self, num_labels, custom_feature_dim):
        super(BertWithCustomFeaturesAndAttention, self).__init__()
        self.bert = BertModel.from_pretrained('from_pretrained/prot_bert_bfd')
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + custom_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask, handcrafted_features):
        # BERT section
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        # Handcrafted feature fusion
        fused_features = torch.cat((sequence_output[:, 0, :], handcrafted_features), dim=1)
        mlp_output = self.mlp(fused_features)
        attention_output, attention_weights = self.attention(mlp_output, mlp_output, mlp_output)
        attention_output = attention_output[:, 0, :]
        logits = self.classifier(attention_output)

        return logits

model=BertWithAttentionAndMLPForSequenceClassification(num_labels).to('cuda')
optimizer = AdamW(model.parameters(), lr=2e-6)
loss_fn = CrossEntropyLoss()


from tqdm import tqdm

highest_accuracy = 0.0
highest_sensitivity = 0.0
highest_specificity = 0.0
highest_auc = 0.0  # add the highest AUC value
for epoch in range(100):
    model.train()
    total_loss = 0
    total_correct = 0
    total_count = 0
    all_labels = []
    all_preds = []
    for batch in tqdm(train_data_loader, desc=f'Epoch {epoch + 1}/{100}'):
        input_ids = batch['input_ids'].to('cuda')
    # print(input_ids)
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        avg_loss = loss.item()
        total_loss += avg_loss

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(logits[:, 1].cpu().detach().numpy())

    auc_score = roc_auc_score(all_labels, all_preds)
    print(f'Epoch {epoch + 1}/{100} | Loss: {total_loss / len(train_data_loader)} | Accuracy: {total_correct / total_count} | AUC: {auc_score}')
    # if (epoch+1)%10==0:
    # test 
    model.eval()
    total_correct = 0
    total_count = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        # for batch in tqdm(test_data_loader, desc='Testing'):
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to('cuda')
            # print(input_ids)
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)

            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

            TP += ((preds == 1) & (labels == 1)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits[:, 1].cpu().detach().numpy())
    # test_accuracy = total_correct / total_count
    test_accuracy=(TP+TN)/(TP+TN+FP+FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    mcc = (TP * TN - FP * FN) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + 1e-10)
    auc_score = roc_auc_score(all_labels, all_preds)
    print(f'Test | Accuracy: {test_accuracy} | Sensitivity (Sn): {sensitivity} | Specificity (Sp): {specificity} | AUC: {auc_score} | MCC: {mcc}')

# Record highest accuracy and highest AUC
    if test_accuracy > highest_accuracy:
        highest_accuracy = test_accuracy
        highest_sensitivity = sensitivity
        highest_specificity = specificity
        highest_auc = auc_score
        highest_mcc = mcc
print(f'Highest Test Accuracy: {highest_accuracy} | Highest Test Sensitivity (Sn): {highest_sensitivity} | Highest Test Specificity (Sp): {highest_specificity} | Highest AUC: {highest_auc} | MCC: {highest_mcc}')

