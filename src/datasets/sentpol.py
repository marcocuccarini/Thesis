import re

import pandas as pd
import torch
from sklearn import preprocessing
from transformers import AutoTokenizer
from src.datasets.hyperion_dataset import *


from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

LABELS = [1,0]



import torch
class PolDataset1(torch.utils.data.Dataset):
    

    def __init__(self, df, tokenizer_name,classType=2):
        #fill_null_features(df)
        #df = filter_empty_labels(df)
        #df = twitter_preprocess(df)
      
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encodings = tokenizer(
        df['text'].tolist(),
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        return_tensors="pt")
        self.labels = encode_labels_bin(df).tolist()
        self.encodingsBert = df['emb'].tolist()

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        item['encBe']= self.encodingsBert[idx]
        
        return item

def encode_labels_bin(df):
    le = preprocessing.LabelEncoder()
    le.fit([0,1])
    return le.transform(df['label'])
