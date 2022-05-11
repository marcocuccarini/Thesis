from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch 

from datasets.hyperion_dataset import decode_labels

class BertRep():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('MiBo/RepML')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained('MiBo/RepML').to(self.device)
        self.model.eval()
    
    def predict(self, text:list[str]):
        encoded_text = self.tokenizer(text,
                                    max_length=512,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt"
                                    )
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

                                    
        logits = self.model(input_ids, attention_mask)['logits']
        logits = logits.detach().cpu()
        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)
        return decode_labels(preds)
