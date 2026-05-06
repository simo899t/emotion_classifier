import dataloader as dl
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


# Embedding layer for the tokens
def build_embedding_layer(vocab_size: int,
                           embed_dim: int,
                           pad_idx: int = 0) -> nn.Embedding:
    return nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)



class SelfAttention(nn.Module):
    def __init__(self, d_model, d_key):
        super().__init__()
        # Three separate linear layers for the queries, keys, and values
        self.w_q = nn.Linear(d_model, d_key)
        self.w_k = nn.Linear(d_model, d_key)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.w_q(x)
        v = self.w_v(x)
        k = self.w_k(x)
        
        def attention(Q,K,V):
            return F.softmax((Q @ torch.transpose(K, -2, -1))/sqrt(K.size(dim=-1)), dim=-1) @ V
        
        return attention(q,k,v)
        

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, d_key, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(d_model, d_key) for _ in range(n_heads)])
        self.w_o = nn.Linear(n_heads * d_model, d_model)

    def forward(self, x):
        result = []
        for head in self.heads:
            result.append(head.forward(x))
        result = torch.cat(result,dim=-1)
        return self.w_o(result)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_key, n_heads, mlp_factor=4):
        super().__init__()


        # We need to init two layer norms because they  have parameters
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, d_key, n_heads)
        self.ln2 = nn.LayerNorm(d_model)

        # a feedforward module with one internal hidden layer
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_factor * d_model),
            nn.SiLU(),  # Swish activation function, f(x) = x * sigmoid(x)
            nn.Linear(mlp_factor * d_model, d_model)
        )

    def forward(self, x):
        # pre-norm
        x = self.attn(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x

        # post-norm
        # x = self.ln1(self.attn() + x)
        # x = self.ln2(self.mlp() + x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, embed, d_model=256, d_key=64, n_heads=4, mlp_factor=4, n_layers=2, pad_idx : int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(vocab_size, embed, padding_idx=pad_idx)

        self.transformer_model = nn.Sequential(*[TransformerBlock(d_model, d_key, n_heads, mlp_factor) for _ in range(n_layers)])
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, n_classes))

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.transformer_model(x)
        x = torch.mean(x, -2)
        x = self.final_layer_norm(x)
        x = self.classifier(x)
        return x

# --------- #



def train(model, padded_batch, targets, epochs=60, lr=1e-3, log_interval = 10):
    """
    Minimal training loop for demonstration purposes.

    Uses Adam + CrossEntropyLoss on the full batch (no train/val split).
    Reports loss and accuracy every log_interval epochs.

    Parameters
    ----------
    model        : Transformer
    padded_batch : LongTensor (B × T)
    lengths      : LongTensor (B × 1)
    targets      : LongTensor (B × 1) (0, 1, 2, 3, 4, 5)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    plot_data = torch.zeros(3, epochs)


    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        logits = model(padded_batch) 

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            preds = logits.argmax(dim=1)
            acc   = (preds == targets).float().mean().item()
            print(f"  Epoch {epoch:3d} | loss {loss.item():.4f} | acc {acc:.2f}")
            plot_data[:, epoch-1] = torch.tensor([epoch, loss, acc])

    return plot_data

def get_model(vocab,NUM_CLASS, D_MODEL, D_KEYS,N_HEAD,MLP_FACTOR,NUM_LAYERS):
    transformer_model = TransformerClassifier(len(vocab),NUM_CLASS,D_MODEL,D_KEYS,N_HEAD,MLP_FACTOR,NUM_LAYERS)
    return transformer_model




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
    #print(f"\nPadded batch shape: {encoded_train_corpus.shape}")
    #print(encoded_train_corpus)

    #EMED_DIM = D_models
    EMBED_DIM = 64
    NUM_LAYERS = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    NUM_CLASS = 6
    D_MODEL = 64
    D_KEYS = 32
    N_HEAD = 4
    MLP_FACTOR = 2

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
    
    print(f"\n--- Training Transfomer Netork with lr: {LEARNING_RATE} on {EPOCHS} epocs ---")
    transfomer_model =  TransformerClassifier(len(vocab),NUM_CLASS,EMBED_DIM, D_MODEL,D_KEYS,N_HEAD,MLP_FACTOR,NUM_LAYERS)
    data = train(transfomer_model, encoded_train_corpus, targets, epochs=EPOCHS, lr=LEARNING_RATE, log_interval=1)
    

