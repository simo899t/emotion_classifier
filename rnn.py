import dataloader as dl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Embedding layer for the tokens
def build_embedding_layer(vocab_size: int,
                           embed_dim: int,
                           pad_idx: int = 0) -> nn.Embedding:
    return nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

# Masked mean pooling
def masked_mean_pool(embeddings: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a mask (inputs != 0) to embeddings and mean pools 
    """
    # unsqueeze to a higher dimension for embedding vectors
    mask_expanded = attention_mask.unsqueeze(-1).float()

    # sum each row of encoded tokens and average them out with num tokens from the mask
    sum_emb = (embeddings * mask_expanded).sum(dim=1)        # (B, D)
    token_counts = attention_mask.sum(dim=1, keepdim=True).float()
    masked_mean_pool = sum_emb / token_counts.clamp(min=1e-9) 
    return masked_mean_pool

class TextRNN(nn.Module):
    """
    TextRNN with the following architecture
    -----------
    1. Embedding
    2. Pack padding
    3. 
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
        self.attn = nn.Linear(hidden_dim, 1)   # learns w and b
        self.dropout   = nn.Dropout(p=0.3)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)                          # (B, T, D)

        packed = pack_padded_sequence(emb, lengths,
                                      batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.rnn(packed)                         # keep outputs
        output, _ = pad_packed_sequence(packed_out,
                                        batch_first=True)        # (B, T, H)
        # output = emb
        T_out = output.size(1)
        attention_mask = (input_ids[:, :T_out] != self.pad_idx)  # (B, T)
        pooled = masked_mean_pool(output, attention_mask)
        pooled = self.dropout(pooled)
        return self.fc(pooled)

        # _, hidden = self.rnn(packed)
        # hidden = hidden[-1]
        # hidden = self.dropout(hidden)   # ← add this
        # return self.fc(hidden)


    
def train(model, padded_batch, lengths, targets,
          val_ids=None, val_lengths=None, val_targets=None,
          epochs=60, lr=1e-3, batch_size= 64, use_lengths=True, log_interval=10):
    """
    Minimal training loop for demonstration purposes.

    Uses Adam + CrossEntropyLoss on the full batch.
    Reports train loss/acc and (optionally) val acc every log_interval epochs.

    Parameters
    ----------
    model        : TextRNN
    padded_batch : LongTensor (B × T)
    lengths      : LongTensor (B,)
    targets      : LongTensor (B,) (0..5)
    val_ids      : LongTensor (Bv × Tv) or None
    val_lengths  : LongTensor (Bv,)    or None
    val_targets  : LongTensor (Bv,)    or None
    use_lengths  : set True for TextRNN (passes lengths to forward)
    """
    train_ds = TensorDataset(padded_batch, lengths, targets)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    plot_data = torch.zeros(4, epochs)
    has_val = val_ids is not None and val_targets is not None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for x_batch, len_batch, y_batch in train_dl:
            optimizer.zero_grad()
            logits = model(x_batch, len_batch) if use_lengths else model(x_batch)
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
                    val_logits = model(val_ids, val_lengths) if use_lengths else model(val_ids)
                    val_acc = (val_logits.argmax(dim=1) == val_targets).float().mean().item()

            print(f"  Epoch {epoch:3d} | loss {train_loss:.4f} | "
                  f"train acc {train_acc:.2f} | val acc {val_acc:.2f}")
            plot_data[:, epoch-1] = torch.tensor([epoch, train_loss, train_acc, val_acc])

    return plot_data

def get_model(vocab, embed_dim, hidden_dim, num_layers):
    num_classes = 6
    rnn_model = TextRNN(len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes)
    return rnn_model

def predict(model, sentence: str, vocab: dl.Vocabulary,
            use_lengths: bool = False) -> str:
    """
    Run a single sentence through a trained model and return the predicted label.

    Steps:
      1. Tokenise and encode the sentence.
      2. Add a batch dimension (B=1).
      3. Forward pass → argmax → human-readable label.
    """
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    model.eval()
    with torch.no_grad():
        ids     = vocab.encode([sentence])
        lengths = torch.tensor([(ids != 0).sum().item()])

        logits  = model(ids, lengths)
        pred    = logits.argmax(dim=1).item()

    return labels[pred]

if __name__ == '__main__':
    train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = dl.load_data()
    vocab = dl.Vocabulary()
    vocab.build_vocabulary(train_inputs)
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
    EPOCHS = 200

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
    data = train(rnn_model, encoded_train_corpus, lengths, targets, use_lengths=True, epochs=EPOCHS, lr=LEARNING_RATE, log_interval=1)
    

