from xmlrpc.client import Boolean
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List
from src.models.bert_rep import *


from src.datasets.hyperion_dataset import decode_labels

# It loads the pretrained model for repertoires prediction and the tokenizer, and provides methods to extract the hidden states of
# the model.
class BertRepEnsamble():
    def __init__(self, model_type1, model_type2):
        self.tokenizer1 = AutoTokenizer.from_pretrained(model_type1)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model1 = AutoModelForSequenceClassification.from_pretrained(model_type1).to(self.device)
        self.model1.eval()

        self.tokenizer2 = AutoTokenizer.from_pretrained(model_type2)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model2 = AutoModelForSequenceClassification.from_pretrained(model_type2).to(self.device)
        self.model2.eval()

    def predictMedia1(self, text:List[str]) -> List[str]:
        encoded_text1 = self.tokenizer1(text,
                                    max_length=512,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt"
                                    )
        encoded_text2 = self.tokenizer2(text,
                                    max_length=512,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt"
                                    )

        input_ids1 = encoded_text1['input_ids'].to(self.device)
        attention_mask1 = encoded_text1['attention_mask'].to(self.device)


        input_ids2 = encoded_text2['input_ids'].to(self.device)
        attention_mask2 = encoded_text2['attention_mask'].to(self.device)


        with torch.no_grad():                          
            logits1 = self.model1(input_ids1, attention_mask1, last_hidden_state= True)
            logits2 = self.model2(input_ids2, attention_mask2, last_hidden_state= True)


        return logits1
    
    def predictMedia(self, text:List[str]) -> List[str]:
        encoded_text1 = self.tokenizer1(text,
                                    max_length=512,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt"
                                    )
        encoded_text2 = self.tokenizer2(text,
                                    max_length=512,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt"
                                    )

        input_ids1 = encoded_text1['input_ids'].to(self.device)
        attention_mask1 = encoded_text1['attention_mask'].to(self.device)


        input_ids2 = encoded_text2['input_ids'].to(self.device)
        attention_mask2 = encoded_text2['attention_mask'].to(self.device)


        with torch.no_grad():                          
            logits1 = self.model1(input_ids1, attention_mask1)['logits']
            logits2 = self.model2(input_ids2, attention_mask2)['logits']

        logits=(logits1+logits2)/2
        logits = logits.detach().cpu()
        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)
        return preds



    def last_hidden_state_average(self, text:List[str]) -> List[str]:
        encoded_text = self.tokenizer(text,
                                    max_length=512,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt"
                                    )
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

        with torch.no_grad():                          
            outputs = self.model(input_ids, attention_mask, output_hidden_states= True)
        hs = outputs['hidden_states'][-1].cpu()
        hs = torch.mean(hs, 1) ## tokens average
        hs = torch.mean(hs, 0) ## spans average
        return hs.tolist()

    def hidden_states(self, text:List[str]) -> List[str]:
        encoded_text = self.tokenizer(text,
                                    max_length=512,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt"
                                    )
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

        with torch.no_grad():                          
            outputs = self.model(input_ids, attention_mask, output_hidden_states= True)
        return torch.stack(outputs['hidden_states']).tolist()
    
    def last_hidden_state_concat(self, text:List[str]) -> List[str]:
        encoded_text = self.tokenizer(text,
                                    max_length=512,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt"
                                    )
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

        with torch.no_grad():                          
            outputs = self.model(input_ids, attention_mask, output_hidden_states= True)
        hs = outputs['hidden_states'][-1].cpu()
        hs = hs.flatten(start_dim= 1) ## tokens concat
        hs = torch.mean(hs, 0) ## spans average
        return hs.tolist()

    def four_last_hidden_state_concat(self, text:List[str]) -> List[str]:
        encoded_text = self.tokenizer(text,
                                    max_length=512,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt"
                                    )
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

        with torch.no_grad():                          
            outputs = self.model(input_ids, attention_mask, output_hidden_states= True)
        hs = outputs['hidden_states'][-4:]
        hs = torch.cat(hs, dim=-1) ## layers concat
        hs = torch.mean(hs, 1) ## tokens average
        hs = torch.mean(hs, 0) ## spans average
        return hs.tolist()
    
    def four_last_hidden_state_sum(self, text:List[str]) -> List[str]:
        encoded_text = self.tokenizer(text,
                                    max_length=512,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt"
                                    )
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

        with torch.no_grad():                          
            outputs = self.model(input_ids, attention_mask, output_hidden_states= True)
        hs = outputs['hidden_states'][-4:]
        hs = torch.sum(hs, dim=0) ## 4 layers sum
        hs = torch.mean(hs, 1) ## tokens average
        hs = torch.mean(hs, 0) ## spans average
        return hs.tolist()
    
    def cls_last_hidden_state(self, text:List[str]) -> List[str]:
        encoded_text = self.tokenizer(text,
                                    max_length=512,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt"
                                    )
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

        with torch.no_grad():                          
            outputs = self.model(input_ids, attention_mask, output_hidden_states= True)
        hs = outputs['hidden_states'][-1].cpu()
        #print(hs.shape)
        hs = hs[:,0,:] ## [CLS] hidden states
        #print(hs.shape)
        hs = torch.mean(hs, 0) ## spans average
        #print(hs.shape)
        #.to_list()
        return hs


       

    

