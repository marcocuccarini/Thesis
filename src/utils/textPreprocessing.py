from ast import literal_eval
import yaml
import sys
import random
import os
import re
import string
import json
import emoji
import math
import numpy as np
import pandas as pd
from sklearn import metrics
from bs4 import BeautifulSoup
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AutoTokenizer, BertModel, BertConfig, AutoModel, AdamW, AutoModelForSequenceClassification
import warnings
from sklearn.preprocessing import MultiLabelBinarizer



drop_rap_freq=[643/35401, 5625/35401, 440/35401, 4402/35401, 2758/35401, 1163/35401, 2122/35401, 616/35401, 639/35401, 705/35401, 1395/35401, 1412/35401, 312/35401, 471/35401, 1102/35401, 638/35401, 1168/35401, 1065/35401, 4882/35401, 1343/35401, 1918/35401, 256/35401, 326/35401]
drop_rap_pred=[0.01,0.26,0.36,0.30,0.08,0.18,0.26,0.42,0.68,0.11,0.2,0.02,0.34,0.12,0.27,0.12,0.3,0.46,0.58,0.68,0.35,0.31,0.43]
drop_name=['Dichiarazione di intenti', 'Sancire', 'Giustificazione', 'Commento', 'Giudizio', 'Non risposta', 'Valutazione', 'Possibilità', 'Conferma', 'Implicazione', 'Specificazione', 'Contrapposizione', 'Considerazione', 'Causa', 'Ridimensionamento', 'Deresponsabilizzazione', 'Previsione', 'Generalizzazione', 'Descrizione', 'Opinione', 'Prescrizione', 'Proposta','Anticipazione']
drop_name,drop_rap_freq = zip(*sorted(zip(drop_name, drop_rap_freq)))
drop_rap_freq=list(drop_rap_freq)


punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-",
                 "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 
                 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!':' '}

def clean_text(text):
    '''Clean emoji, Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = emoji.demojize(text)
    text = re.sub(r'\:(.*?)\:','',text)
    text = str(text).lower()    #Making Text Lowercase
    text = re.sub('\[.*?\]', '', text)
    #The next 2 lines remove html text
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    return text

def clean_contractions(text, mapping):
    '''Clean contraction using contraction mapping'''    
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    for word in mapping.keys():
        if ""+word+"" in text:
            text = text.replace(""+word+"", ""+mapping[word]+"")
    #Remove Punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text

def clean_special_chars(text, punct, mapping):
    '''Cleans special characters present(if any)'''   
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

def correct_spelling(x, dic):
    '''Corrects common spelling errors'''   
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

def remove_space(text):
    '''Removes awkward spaces'''   
    #Removes awkward spaces 
    text = text.strip()
    text = text.split()
    return " ".join(text)

def text_preprocessing_pipeline(text):
    '''Cleaning and parsing the text.'''
    text = clean_text(text)
    #text = clean_contractions(text, contractions_dict)
    text = clean_special_chars(text, punct, punct_mapping)
    text = remove_space(text)
    return text


def addEncSimple(df, test_df,dropout):
  from ast import literal_eval
  df['Repertori_predetti'] = np.array(df['Repertori_predetti'].apply(literal_eval))
  test_df['Repertori_predetti'] = np.array(test_df['Repertori_predetti'].apply(literal_eval))

  one_hot = MultiLabelBinarizer()
  enc=one_hot.fit_transform(df['Repertori_predetti'])
  enc_test=one_hot.fit_transform(test_df['Repertori_predetti'])

  enc=np.array(enc)
  enc_test=np.array(enc_test)
  listString=[]
  for i in enc:
      cont=0
      string=" "
      for j in i:
        r=random.random()
        
        if(r<(1-dropout)):
          string+=" "
          string+=str(j)
        else:
          string+=""
          string+=""
        cont+=1
          

      listString.append(string)
  listStringtest=[]
  for i in enc_test:
      cont=0
      string=" "
      for j in i:
        r=random.random()
        if(r<(1-dropout)):
            string+=" "
            string+=str(j)
        else:
          string+=""
          string+=""
        cont+=1

      listStringtest.append(string)
  df['text']=(df['text']+listString)
  test_df['text']=(test_df['text']+listStringtest)

  return df, test_df

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def addEncComp(df, test_df, position,dropout,drop_index):
  sigmoid_v = np.vectorize(sigmoid)

  df['Rep Out'] = df['Rep Out'].apply(literal_eval)
  test_df['Rep Out'] = test_df['Rep Out'].apply(literal_eval)
  listString=[]


  enc=sigmoid_v(np.array(df['Rep Out'].to_list()))
  enc_test=sigmoid_v(np.array(test_df['Rep Out'].to_list()))
  #enc=(np.array(df['Rep Out'].to_list()))
  #enc_test=(np.array(test_df['Rep Out'].to_list()))

  for i in enc:
      string=" "
      cont=0
      for j in i[0]:
        
        r=random.random()
        if(r<(1-dropout*drop_rap_freq[cont]*drop_index)):
          string+=" "
          string+=str(j)
        else:
          string+=" "
          string+="0"
        cont+=1
      listString.append(string)
  listStringtest=[]
  for i in enc_test:
      string=" "
      cont=0
      for j in i[0]:
        r=random.random()
        if(r<(1-dropout*drop_rap_pred[cont]*drop_index)):
          string+=" "
          string+=str(j)

        else:
          string+=" "
          string+="0"
        cont+=1
      listStringtest.append(string)
  if position:
    df['text']=np.array(df['text']+listString)
    test_df['text']=np.array(test_df['text']+listStringtest)
  else:
    df['text']=np.array(+listString+df['text'])
    test_df['text']=np.array(listStringtest+test_df['text'])

  return df, test_df





