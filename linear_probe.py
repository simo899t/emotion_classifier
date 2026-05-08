import dataloader as dl
import rnn
import lstm
import model_2 as m2
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



class LinearProbe(nn.Module):
    def __init__(self):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(in_features=64, out_features=2)


    def forward(self, x):
        x = x.view(-1, 1)
        return fc(x)









if __name__ == "__main__":
    train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = dl.load_data()
    vocab = dl.Vocabulary()
    vocab.build_vocabulary(train_inputs)
    encoded_train_corpus = vocab.encode(train_inputs)
    train_targets = torch.tensor(train_labels)
    train_lengths = (encoded_train_corpus != 0).sum(dim=1)   # LongTensor (B × 1)

    encoded_val = vocab.encode(val_inputs)
    val_lengths = (encoded_val != 0).sum(dim=1)
    val_targets = torch.tensor(val_labels)
    val_targets[val_targets != 1] = 0 # Everything that is not joy, should just be "wrong"


    EMBED_DIM = 128
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    BATCH_SIZE = 128
    
    rnn_model = rnn.get_model(vocab, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS)
    use_lengths=True
    """
    print(f"\n--- Training TextRNN with lr: {LEARNING_RATE} on {EPOCHS} epocs ---")
    T1 = rnn.train(rnn_model, encoded_train_corpus, train_lengths, train_targets,
                  val_ids=encoded_val, val_lengths=val_lengths, val_targets=val_targets,
                  use_lengths=True, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, log_interval=1)
    """

    rnn_model.eval()
    train_ds = TensorDataset(encoded_train_corpus, train_lengths, train_targets)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    probe = LinearProbe()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(probe.parameters(), lr=0.001)
    
    for epoch in range(1, EPOCHS + 1):
        probe.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for x_batch, len_batch, y_batch in train_dl:
            with torch.no_grad():
                _, hidden_layers = rnn_model(x_batch, len_batch) if use_lengths else model(x_batch)
                final_hidden_layer = hidden_layers[-1, :, :]


            optimizer.zero_grad()
            loss = criterion(final_hidden_layer.squeeze(), y_batch)
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
                    val_logits = model(val_ids, val_lengths) if use_lengths else model(val_ids)
                    val_acc = (val_logits.argmax(dim=1) == val_targets).float().mean().item()

            print(f"  Epoch {epoch:3d} | loss {train_loss:.4f} | "
                  f"train acc {train_acc:.2f} | val acc {val_acc:.2f}")



    













