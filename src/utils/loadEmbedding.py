import random
import os

import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from src.models.bert_rep import *



def getEmbedding(pathTest,pathTrain,name):
  
  train_Alb_Em = np.load(pathTrain,allow_pickle=True,)
  test_Alb_Em = np.load(pathTest,allow_pickle=True,)

  train_df = pd.read_csv('data/processed/sentipolcAlBERTo/TrainAlBERToSentiPolc.csv')
  test_df = pd.read_csv('data/processed/sentipolcAlBERTo/TestAlBERToSentiPolc.csv')

  train_df['emb'+name]=train_Alb_Em.tolist()
  test_df['emb'+name]=test_Alb_Em.tolist()

  return train_df,test_df
#Deterministic mode
def dowloadEmbedding(BERT,AlBERTo,BERT3,BERT3_Un,MultiBERT,BERTCased,Electra,TypeEmb,Head=0): 
    embeddings_method=['cls_last_hidden_state','last_hidden_state_average','last_hidden_state_concat','four_last_hidden_state_concat','four_last_hidden_state_sum']

    train_df = pd.read_csv('data/processed/sentipolcAlBERTo/TrainAlBERToSentiPolc.csv')
    test_df = pd.read_csv('data/processed/sentipolcAlBERTo/TestAlBERToSentiPolc.csv')
    
    if (Head):
        train_df=train_df.head(Head)
        test_df=test_df.head(Head)

    if(BERT):
        bert_rep=BertRep("MiBo/RepML")
        embeddings_method_BERT = getattr(bert_rep, embeddings_method[TypeEmb])
        train_df['embBERT'] = train_df['Stralci_predetti'].map(embeddings_method_BERT).values.tolist()
        test_df['embBERT'] = test_df['Stralci_predetti'].map(embeddings_method_BERT).values.tolist()

    if(AlBERTo):
        AlBERTo_rep = BertRep("Marco127/DRAlBERTo")
        embeddings_method_AlBERTo = getattr(AlBERTo_rep, embeddings_method[TypeEmb])
        train_df['embAlBERT'] = train_df['Stralci_predetti'].map(embeddings_method_AlBERTo).values.tolist()
        test_df['embAlBERT'] = test_df['Stralci_predetti'].map(embeddings_method_AlBERTo).values.tolist()

    
    if(BERT3):
        bert_rep_3 = BertRep("Marco127/BERTUnCasedColor")
        embeddings_method_BERT_3 = getattr(bert_rep_3, embeddings_method[TypeEmb])
        train_df['embBERT_3'] = train_df['Stralci_predetti'].map(embeddings_method_BERT_3).values.tolist()
        test_df['embBERT_3'] = test_df['Stralci_predetti'].map(embeddings_method_BERT_3).values.tolist()
    
    if(BERT3_Un):
        bert_rep_3_unbalanced= BertRep("Marco127/AlBERToColorUnbalanced")
        embeddings_method_BERT_3_Unbalced = getattr(bert_rep_3_unbalanced, embeddings_method[TypeEmb])
        train_df['embBERT_3_Un'] = train_df['Stralci_predetti'].map(embeddings_method_BERT_3_Unbalced).values.tolist()
        test_df['embBERT_3_Un'] = test_df['Stralci_predetti'].map(embeddings_method_BERT_3_Unbalced).values.tolist()
    
    if(MultiBERT):
        multiBERT_rep= BertRep("Marco127/MultiBert")
        embeddings_method_MultiBERT = getattr(multiBERT_rep, embeddings_method[TypeEmb])
        train_df['embMultiBERT'] = train_df['Stralci_predetti'].map(embeddings_method_MultiBERT).values.tolist()
        test_df['embMultiBERT'] = test_df['Stralci_predetti'].map(embeddings_method_MultiBERT).values.tolist()
    if(BERTCased):
        CasedBERT_rep= BertRep("Marco127/BERTCased")
        embeddings_method_BERTCased = getattr(CasedBERT_rep, embeddings_method[TypeEmb])
        train_df['embCasedBERT'] = train_df['Stralci_predetti'].map(embeddings_method_BERTCased).values.tolist()
        test_df['embCasedBERT'] = test_df['Stralci_predetti'].map(embeddings_method_BERTCased).values.tolist()
    if(Electra):
        Electra_rep= BertRep("Marco127/Electra")
        embeddings_method_Electra = getattr(Electra_rep, embeddings_method[TypeEmb])
        train_df['embElectra'] = train_df['Stralci_predetti'].map(embeddings_method_Electra).values.tolist()
        test_df['embElectra'] = test_df['Stralci_predetti'].map(embeddings_method_Electra).values.tolist()
    return train_df,test_df

def modelComposition(A,B,C,D,train_df,test_df,ModelA,ModelB,ModelC,ModelD,Task,X_train,X_test):
    X_train=[[((i*A)+(j*B)+(z*C)+(D*h))/(A+B+C+D) for i, j, z, h in zip(x, y , o, k)] for x, y, o, k  in zip( train_df["emb"+ModelA].to_list(),train_df["emb"+ModelB].to_list(),train_df["emb"+ModelC].to_list(),train_df["emb"+ModelD].to_list())]
    X_test=[[((i*A)+(j*B)+(z*C)+(D*h))/(A+B+C+D) for i, j, z, h in zip(x, y, o, k)] for x, y, o, k in zip(test_df["emb"+ModelA].to_list(),test_df["emb"+ModelB].to_list(),test_df["emb"+ModelC].to_list(),test_df["emb"+ModelD].to_list())]
    y_train=train_df[Task].to_list()
    y_test=test_df[Task].to_list()
    return X_train,X_test,y_train,y_test