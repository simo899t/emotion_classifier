
import dataloader as dl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader

from transformers import (
    BertTokenizer, DistilBertTokenizer, 
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import torch 

import torch.nn as nn
from transformers import BertTokenizer, BertModel, DistilBertModel

# Bert tokenizer 

# Freeze pretrained BERT layers, so the semantic meaning of the embedding space don't change

def freeze_bert(bert):
        for param in bert.parameters():
            param.requires_grad = False

class BertClassifier(nn.Module):
    def __init__(self, d_features, n_classes):
        super().__init__()
        self.bert =self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        freeze_bert(self.bert)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x, mask):
        output = self.bert(x, attention_mask=mask)
        #pooled_output = output.pooler_output
        cls_output = output.last_hidden_state[:, 0, :]
        x = self.fc1(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)



tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def train(model, padded_batch, targets, attention_mask ,use_lengths = True,
          val_ids=None, val_mask = None, val_targets=None,
          epochs=60, lr=1e-3, batch_size=4, log_interval = 10):
    """
    Minimal training loop for demonstration purposes.

    Uses Adam + CrossEntropyLoss on the full batch (no train/val split).
    Reports loss and accuracy every log_interval epochs.

    Parameters
    ----------
    model        : Transformer
    padded_batch : LongTensor (B × T)
    targets      : LongTensor (B × 1) (0, 1, 2, 3, 4, 5)
    """

    train_ds = TensorDataset(padded_batch, attention_mask, targets)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    plot_data = torch.zeros(4, epochs)
    has_val = val_ids is not None and val_targets is not None


    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for x_batch, len_batch, y_batch in train_dl:
            optimizer.zero_grad()
            logits = model(x_batch, len_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss    += loss.item() * y_batch.size(0)
            running_correct += (logits.argmax(1) == y_batch).sum().item()
            running_total   += y_batch.size(0)

        train_loss = running_loss / running_total
        train_acc  = running_correct / running_total

        if epoch % log_interval == 0:
            val_acc = float("nan")
            if has_val:
                model.eval()
                with torch.no_grad():
                    val_logits = model(val_ids, val_mask)
                    val_acc = (val_logits.argmax(dim=1) == val_targets).float().mean().item()

            print(f"  Epoch {epoch:3d} | loss {train_loss:.4f} | "
                  f"train acc {train_acc:.2f} | val acc {val_acc:.2f}")
            plot_data[:, epoch-1] = torch.tensor([epoch, train_loss, train_acc, val_acc])

    return plot_data


def get_model(n_features, n_classes):
    num_classes = n_classes
    num_classes = 6 
    bert_model = BertClassifier(n_features, num_classes)
    return bert_model




N_FEATURES = 768
NUM_LAYERS = 4
LEARNING_RATE = 3e-5
EPOCHS = 3
NUM_CLASS = 6

if __name__ == '__main__':
    # Bert tokenizer 
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = dl.load_data()
    encoded_train_corpus = tokenizer(train_inputs,padding=True, truncation=True, 
                    max_length=32, return_tensors='pt')
    encoded_val_corpus = tokenizer(val_inputs,padding=True, truncation=True, 
                    max_length=32, return_tensors='pt')
    encoded_test_corpus = tokenizer(test_inputs,padding=True, truncation=True, 
                    max_length=32, return_tensors='pt')


    train_seq = torch.tensor(encoded_train_corpus['input_ids'])
    train_mask = torch.tensor(encoded_train_corpus['attention_mask'])
    train_y = torch.tensor(train_labels)

    val_seq = torch.tensor(encoded_val_corpus['input_ids'])
    val_mask = torch.tensor(encoded_val_corpus['attention_mask'])
    val_y = torch.tensor(val_labels)

    test_seq = torch.tensor(encoded_test_corpus['input_ids'])
    test_mask = torch.tensor(encoded_test_corpus['attention_mask'])
    test_y = torch.tensor(test_labels)


   
    

    print(f"\n--- Training Transfomer Netork with lr: {LEARNING_RATE} on {EPOCHS} epocs ---")
    Bert_model =  get_model(N_FEATURES,NUM_CLASS)
    data = train(Bert_model, train_seq, train_y, train_mask,
             val_ids=val_seq, val_mask=val_mask, val_targets=val_y,
             epochs=EPOCHS, lr=LEARNING_RATE)


