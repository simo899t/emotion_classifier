import dataloader as dl
import rnn
import lstm
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



class LinearProbe(nn.Module):
    def __init__(self, hidden_dim):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

def train_probe(model, probe, padded_batch, lengths, targets, epochs, lr, batch_size):
    model.eval()
    train_ds = TensorDataset(padded_batch, lengths, targets)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(probe.parameters(), lr=0.001)
    total_loss = []

    for epoch in range(1, epochs + 1):
        probe.train()

        for x_batch, len_batch, y_batch in train_dl:
            with torch.no_grad():
                _, hidden_layers = trained_rnn(x_batch, len_batch)

            optimizer.zero_grad()
            outputs = probe(hidden_layers)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

        print(loss.item())
        total_loss.append(loss.item())
    return total_loss, model

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

    probe_train_targets = (train_targets == 1).float()

    EMBED_DIM = 128
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    PROBE_EPOCHS = 10
    BATCH_SIZE = 32
    
    rnn_model = rnn.get_model(vocab, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS)
    print(f"\n--- Training TextRNN with lr: {LEARNING_RATE} on {EPOCHS} epocs ---")
    T1, trained_rnn = rnn.train(rnn_model, encoded_train_corpus, train_lengths, train_targets,
                                val_ids=encoded_val, val_lengths=val_lengths, val_targets=val_targets,
                  use_lengths=True, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE)

    probe = LinearProbe(HIDDEN_DIM)
    total_loss, trained_probe = train_probe(trained_rnn, probe, encoded_train_corpus, train_lengths, probe_train_targets, 
                                            epochs=PROBE_EPOCHS, lr=LEARNING_RATE,batch_size=BATCH_SIZE)
    print(total_loss)
    f, ax1 = plt.subplots(1,1)
    x = range(1, PROBE_EPOCHS+1)
    ax1.plot(x, total_loss, label='Probe')
    ax1.set_title('Loss')
    ax1.legend()
    plt.tight_layout()
    plt.show()






    













