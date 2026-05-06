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
    def __init__(self, vocab_size, embed_dim, n_classes, d_model=256, d_key=64, n_heads=4, mlp_factor=4, n_layers=2, pad_idx : int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(vocab_size, embed_dim,padding_idx=pad_idx)

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

if __name__ == '__main__':

    train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = dl.load_data()

    vocab = dl.Vocabulary()
    vocab.build_vocabulary(train_inputs)
    encoded_train_corpus = vocab.encode(train_inputs)
    encoded_val_corpus = vocab.encode(val_inputs)

    # decoded = vocab.decode(vocab.encode())

    # Set seeds
    torch.manual_seed(0)

    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # print(f'Using device: {device}')

    # Create the model

    # Validation accuracy: 98.50000143051147%



    
    EMBED_DIM = 64
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 1e-2
    EPOCHS = 100
    NUM_CLASS = 6 
    D_MODEL = 256
    D_KEYS = 64
    N_HEAD = 4
    MLP_FACTOR = 4 

    model = TransformerClassifier(len(vocab),EMBED_DIM,NUM_CLASS,D_MODEL,D_KEYS,N_HEAD,MLP_FACTOR,NUM_LAYERS)


    n_epochs = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(encoded_train_corpus)
        loss = criterion(y_pred, train_labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    # Check the validation accuracy
    with torch.no_grad():
        model.eval()
        y_pred = model(encoded_val_corpus)
        acc = (torch.argmax(y_pred, dim=1) == val_labels).float().mean()
        print(f'Validation accuracy: {100*acc.item()}%')

    # Number of parameters
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')



