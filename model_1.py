import dataloader as dl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence


# Embedding layer for the tokens
def build_embedding_layer(vocab_size: int,
                           embed_dim: int,
                           pad_idx: int = 0) -> nn.Embedding:
    return nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

class TextRNN(nn.Module):
    """
    Sequential text classifier built around a single-layer RNN.

    Architecture:
      Embedding (frozen PAD)
        → pack_padded_sequence      [skip PAD positions during recurrence]
        → RNN
        → pad_packed_sequence       [restore (B, T, H) layout]
        → final hidden state h_T    [shape (B, H)]
        → Linear(hidden_dim, num_classes)
        → raw logits

    Why use the final hidden state instead of mean pooling?
    -------------------------------------------------------
    The RNN's hidden state is updated at every real token, so h_T is a
    recurrent summary that is sensitive to word order and context.

    Why pack/unpack?
    ----------------
    pack_padded_sequence tells the RNN exactly how many real time steps each
    example has, so it never feeds <PAD> positions into the recurrence. This
    avoids contaminating the final state for shorter sequences.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_classes: int,
                 pad_idx: int = 0):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=pad_idx)
        # batch_first=True  →  input/output tensors are (B, T, *)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self,
                input_ids: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : LongTensor (B, T)   — padded token indices
        lengths   : LongTensor (B,)     — real (non-PAD) length of each sentence

        Returns
        -------
        FloatTensor (B, num_classes) — raw class scores
        """
        # encoded_token * v. So that (B, T) → (B, T, D)
        emb = self.embedding(input_ids)

        # use lengths of each sentence, to ignore zeroes when training
        packed = pack_padded_sequence(emb, lengths,
                                      batch_first=True,
                                      enforce_sorted=False)

        # ignore all packed hidden states, we only care about the last hidden layer
        _, hidden = self.rnn(packed)
        
        # 4. (Optional) unpack output if you need all hidden states:
        #    output, _ = pad_packed_sequence(_output, batch_first=True)
        #    — not needed when using only the final hidden state.

        # 5. Take the last layer's hidden state: (num_layers, B, H) → (B, H)
         
        hidden = hidden[-1]
         

        # 6. Linear classifier  (B, H) → (B, num_classes)
        return self.fc(hidden)
    
def train(model, padded_batch, lengths, targets, epochs=60, lr=1e-3, use_lengths=True, log_interval = 10):
    """
    Minimal training loop for demonstration purposes.

    Uses Adam + CrossEntropyLoss on the full batch (no train/val split).
    Reports loss and accuracy every log_interval epochs.

    Parameters
    ----------
    model        : TextRNN
    padded_batch : LongTensor (B × T)
    lengths      : LongTensor (B × 1)
    targets      : LongTensor (B × 1) (0, 1, 2, 3, 4, 5)
    use_lengths  : set True for TextRNN (passes lengths to forward)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        logits = model(padded_batch, lengths) if use_lengths else model(padded_batch)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            preds = logits.argmax(dim=1)
            acc   = (preds == targets).float().mean().item()
            print(f"  Epoch {epoch:3d} | loss {loss.item():.4f} | acc {acc:.2f}")

if __name__ == '__main__':
    train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = dl.load_data()
    vocab = dl.Vocabulary()
    vocab.build_vocabulary(test_inputs)
    encoded_train_corpus = vocab.encode(train_inputs)

    #print("\n" + "=" * 60)
    #print("PART 2 — Vocabulary")
    #print("=" * 60)
    #print(f"Vocabulary size : {len(vocab)}")
    #print(f"<PAD> index     : {vocab.token2idx['<PAD>']} (should be 0)")
    #print(f"<UNC> index     : {vocab.token2idx['<UNC>']} (should be 1)")
    #print(f"Encoded sent[0] : {encoded_train_corpus}")
    #print(f"Decoded back    : {vocab.decode(encoded_train_corpus)}")
    
    # Test OOV handling
    #print(f"Unknown token   : {vocab.encode(['notaword'])} (should be [1])")
    #print(f"\nPadded batch shape : {encoded_train_corpus.shape}")
    #print(encoded_train_corpus)

    EMBED_DIM = 64
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 1e-2
    EPOCHS = 100

    embedding_layer = build_embedding_layer(len(vocab), EMBED_DIM)

    # <PAD> embedding must be all zeros
    pad_emb = embedding_layer(torch.tensor([0]))
    print(f"<PAD> embedding all-zero: {pad_emb.abs().sum().item() == 0.0}")
    
    # Embed the whole padded batch
    embedded = embedding_layer(encoded_train_corpus)   # (B × T × D)
    print(f"Embedded batch shape    : {embedded.shape}  (B=6, T=6, D={EMBED_DIM})")

    # Compute real sequence lengths (count of non-PAD tokens per row)
    lengths = (encoded_train_corpus != 0).sum(dim=1)   # LongTensor (B × 1)

    targets = torch.tensor(train_labels)
    num_classes = 6      # works regardless of label values
    
    print(f"\n--- Training TextRNN with lr: {LEARNING_RATE} on {EPOCHS} epocs ---")
    rnn_model = TextRNN(len(vocab), EMBED_DIM, hidden_dim=32, num_layers=NUM_LAYERS, num_classes=num_classes)
    train(rnn_model, encoded_train_corpus, lengths, targets, use_lengths=True, epochs=EPOCHS, lr=LEARNING_RATE, log_interval=1)

