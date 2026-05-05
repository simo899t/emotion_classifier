import dataloader as dl
import torch
import torch.nn as nn

# Embedding
   # dimensionality of token embeddings

def build_embedding_layer(vocab_size: int,
                           embed_dim: int,
                           pad_idx: int = 0) -> nn.Embedding:
    return nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)





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

    EMBED_DIM = 32

    embedding_layer = build_embedding_layer(len(vocab), EMBED_DIM)

    # <PAD> embedding must be all zeros
    pad_emb = embedding_layer(torch.tensor([0]))
    print(f"<PAD> embedding all-zero: {pad_emb.abs().sum().item() == 0.0}")
    
    # Embed the whole padded batch
    embedded = embedding_layer(encoded_train_corpus)   # (B, T, D)
    print(f"Embedded batch shape    : {embedded.shape}  (B=6, T=6, D={EMBED_DIM})")
