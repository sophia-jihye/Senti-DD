from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch, os

def train(model, train_dataset, val_dataset, save_dir):
    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        save_steps = 10000,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print('[Fine-tuning] Training..')
    trainer.train()

    trainer.save_model(save_dir)
    print('[Fine-tuning] Saved trained model at {}'.format(save_dir))

def inference(model, input_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        try: outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
        except: return None
    logits = outputs['logits'].detach().cpu().numpy()
    predicted_label = np.argmax(logits)
    return predicted_label