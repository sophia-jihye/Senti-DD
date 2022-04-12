from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM, AutoModel
from transformers import pipeline
import numpy as np

from CustomDataset import encode_for_inference

def load_tokenizer_and_model(model_name_or_path, num_classes=None, mode='classification'):
    if num_classes is not None: # train
        config = AutoConfig.from_pretrained(model_name_or_path, num_classes=num_classes)
    else: # test
        config = AutoConfig.from_pretrained(model_name_or_path)      
    
    print('Loading tokenizer & model for {}..\n'.format(model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if mode == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    elif mode == 'masking':
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
        
    return tokenizer, model

class FeatureExtractor:
    def __init__(self, model_name_or_dir, device):
        self.tokenizer, self.model = load_tokenizer_and_model(model_name_or_dir)
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def get_cls_embedding(self, text):
        input_ids, attention_mask = encode_for_inference(self.device, self.tokenizer, text)
        cls_feature = self.model(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)['hidden_states'][0] # Number of layers: 13   (initial [CLS] embeddings + 12 BERT layers)
        return np.squeeze(cls_feature).mean(0).detach().cpu().numpy() # 평균을 취하는 방법을 택함

# 주어진 문서의 BERT embedding 값을 구할 때 사용
class FeatureExtractor_pipeline:
    def __init__(self, pretrained_model_name):
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.nlp = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer)
    
    def get_feature(self, text):
        try:
            feature = self.nlp(text)
        except:
            text = self.tokenizer.decode(self.tokenizer(text, truncation=True, padding='max_length', max_length=512))
            text = text.replace('[CLS]', '').replace('[SEP]', '') 
            feature = self.nlp(text)
            
        feature = np.squeeze(feature)
        return feature.mean(0) # 평균을 취하는 방법을 택함