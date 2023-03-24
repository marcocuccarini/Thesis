from xmlrpc.client import Boolean
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List

from src.datasets.hyperion_dataset import decode_labels

# It loads the pretrained model for repertoires prediction and the tokenizer, and provides methods to extract the hidden states of
# the model.
class BertPol():
    def __init__(self, model_type):
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_type).to(self.device)
        self.model.eval()
    
    def predictEmb(self, encoded_text,attention_mask) -> List[str]:

        input_ids = encoded_text.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():                          
            logits = self.model(input_ids, attention_mask)['logits']
        logits = logits.detach().cpu()
        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)
        return preds
    
    

    

