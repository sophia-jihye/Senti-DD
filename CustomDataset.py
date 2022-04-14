from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, tokenizer, raw_texts, labels):
        self.tokenizer = tokenizer
        self.raw_texts = raw_texts
        self.labels = labels

    def __getitem__(self, idx):
        text = self.raw_texts[idx]
        item = {key: torch.tensor(val) for key, val in self.tokenizer(text, truncation=True, padding='max_length', max_length=512).items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long) 
        item['raw_text'] = text
        return item

    def __len__(self):
        return len(self.labels)
    
def encode_for_inference(device, tokenizer, text):
    data = {key: torch.tensor(val) for key, val in tokenizer(text, truncation=True, padding='max_length', max_length=512).items()}
    return torch.unsqueeze(data['input_ids'], 0).to(device), torch.unsqueeze(data['attention_mask'], 0).to(device)