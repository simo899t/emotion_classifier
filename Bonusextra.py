import dataloader as dl
import torch
import numpy as np
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score

# Load data
train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = dl.load_data()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Creates dictionary from our data, since Trainer need a Dataset Object
train_dataset = Dataset.from_dict({'text': train_inputs, 'labels': train_labels})
print(train_dataset)
val_dataset   = Dataset.from_dict({'text': val_inputs,   'labels': val_labels})
test_dataset  = Dataset.from_dict({'text': test_inputs,  'labels': test_labels})

# Tokenize
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=69)

#Remove plain text and only have inters and values 
#Label, numberical representation and attention mask as strings

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['text'])
val_dataset   = val_dataset.map(tokenize,   batched=True, remove_columns=['text'])
test_dataset  = test_dataset.map(tokenize,  batched=True, remove_columns=['text'])

#Setting our data object to tensor 
train_dataset.set_format('torch')
val_dataset.set_format('torch')
test_dataset.set_format('torch')

# Bert model build with classification module MLP.
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=6
)

# To compute test Accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Training setting
training_args = TrainingArguments(
    output_dir='bert-emotion-finetuned', #Our folder Trainer saves checkpoints and model files during training
    num_train_epochs=3, 
    per_device_train_batch_size=64, #Batch for training 
    per_device_eval_batch_size=32, #Batch size for Validation and test
    learning_rate= 0.002, #Learning rate should be small says the internet
    weight_decay=0.01, #Weight_decay
    logging_steps=100, #Update her 100 steps
    eval_strategy="epoch", #evaluation during training
    save_strategy="epoch", #Load the best checkpoint at the end of training
    load_best_model_at_end=True, #
)

# Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)


trainer.train()


results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test accuracy: {results['eval_accuracy']:.4f}")

