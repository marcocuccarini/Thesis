from xmlrpc.client import Boolean
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List

from datasets.hyperion_dataset import decode_labels

class BertRep():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('MiBo/RepML')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained('MiBo/RepML').to(self.device)
        self.model.eval()
    
    def predict(self, text:List[str]) -> List[str]:
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
            logits = self.model(input_ids, attention_mask)['logits']
        logits = logits.detach().cpu()
        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)
        return decode_labels(preds).tolist()
    
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
    

    

