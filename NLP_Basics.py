"""
nlp_pipeline.py
===============
University exercise: NLP fundamentals in PyTorch.

Covers:
  Part 1 — Tokenization   (whitespace and character-level)
  Part 2 — Vocabulary     (token↔index mapping + padding)
  Part 3 — Embeddings     (nn.Embedding with frozen <PAD> vector)
  Part 4 — TextMLP        (mean pooling → two-layer MLP)
  Part 5 — TextRNN        (RNN with packed sequences)

Run with:
  python nlp_pipeline.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from datasets import load_dataset


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Dataset
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

sentences = [
    "the film was wonderful and touching",
    "terrible movie very boring and slow",
    "great performances and a beautiful story",
    "awful acting the plot made no sense",
    "loved every moment of this masterpiece",
    "waste of time poorly written script",
]
labels = [1, 0, 1, 0, 1, 0]   # 1 = positive review, 0 = negative review


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Tokenization
# ═══════════════════════════════════════════════════════════════════════════════

def whitespace_tokenize(text: str, max_len: int = 100) -> list[str]:
    """
    Lowercase the string and split on any whitespace.

    This is the simplest possible tokenizer: every run of non-space characters
    becomes one token.  Fast and interpretable, but it produces a large
    vocabulary that grows with corpus size and cannot handle unseen words.

    Example
    -------
    >>> whitespace_tokenize("The film was GREAT")
    ['the', 'film', 'was', 'great']
    """
    return text.lower().split()[:max_len]


def char_tokenize(text: str) -> list[str]:
    """
    Lowercase the string and return one token per character, excluding spaces.

    Character-level tokenisation gives a tiny, fixed vocabulary (≈50–100
    types) and is immune to out-of-vocabulary words.  The trade-off is that
    sequences become ~5× longer than word-level sequences, making it harder
    for a model to learn long-range semantic relationships.

    Example
    -------
    >>> char_tokenize("Hi NLP")
    ['h', 'i', 'n', 'l', 'p']
    """
    return [ch for ch in text.lower() if ch != " "]


# ── Quick smoke tests ──────────────────────────────────────────────────────────
print("=" * 60)
print("PART 1 — Tokenization")
print("=" * 60)
print("whitespace:", whitespace_tokenize("The film was GREAT"))
print("char-level:", char_tokenize("Hi NLP"))


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

class Vocabulary:
    """
    Maps tokens to integer indices and back.

    Two special tokens are always present:
      <PAD>  (index 0) — used to pad shorter sequences to a common length.
                         Its embedding vector is frozen at zero.
      <UNK>  (index 1) — replaces any token that was not seen during build().

    All other tokens are assigned consecutive indices in the order they are
    first encountered while iterating over the corpus.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}

    # ------------------------------------------------------------------
    def build(self, tokenized_corpus: list[list[str]]) -> None:
        """
        Populate the vocabulary from a list of already-tokenised sentences.

        Special tokens are inserted first so their indices are always 0 and 1,
        regardless of whether they appear in the corpus text.
        """
        # Insert special tokens with fixed indices
        for special in (self.PAD_TOKEN, self.UNK_TOKEN):
            idx = len(self.token2idx)
            self.token2idx[special] = idx
            self.idx2token[idx] = special

        # Add every new token in encounter order
        for sentence in tokenized_corpus:
            for token in sentence:
                if token not in self.token2idx:
                    idx = len(self.token2idx)
                    self.token2idx[token] = idx
                    self.idx2token[idx] = token

    # ------------------------------------------------------------------
    def encode(self, tokens: list[str]) -> list[int]:
        """
        Convert a list of token strings to a list of integer indices.
        Unknown tokens are mapped to the <UNK> index (1).
        """
        unk_idx = self.token2idx[self.UNK_TOKEN]
        return [self.token2idx.get(t, unk_idx) for t in tokens]

    # ------------------------------------------------------------------
    def decode(self, indices: list[int]) -> list[str]:
        """Convert a list of indices back to token strings."""
        return [self.idx2token[i] for i in indices]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.token2idx)


# ── Build and inspect ─────────────────────────────────────────────────────────
tokenized_corpus = [whitespace_tokenize(s) for s in sentences]
vocab = Vocabulary()
vocab.build(tokenized_corpus)

print("\n" + "=" * 60)
print("PART 2 — Vocabulary")
print("=" * 60)
print(f"Vocabulary size : {len(vocab)}")
print(f"<PAD> index     : {vocab.token2idx['<PAD>']} (should be 0)")
print(f"<UNK> index     : {vocab.token2idx['<UNK>']} (should be 1)")
sample_enc = vocab.encode(tokenized_corpus[0])
print(f"Encoded sent[0] : {sample_enc}")
print(f"Decoded back    : {vocab.decode(sample_enc)}")

# Test OOV handling
print(f"Unknown token   : {vocab.encode(['notaword'])} (should be [1])")


# ── Padding ───────────────────────────────────────────────────────────────────

def pad_sequences(sequences: list[list[int]], pad_idx: int = 0) -> torch.Tensor:
    """
    Pad a batch of variable-length index sequences to the same length.

    All sequences are right-padded with pad_idx until they match the length
    of the longest sequence in the batch.

    Parameters
    ----------
    sequences : list of int lists, each of potentially different length
    pad_idx   : the index used for padding (should match <PAD> in Vocabulary)

    Returns
    -------
    torch.LongTensor of shape (batch_size, max_len)

    Example
    -------
    pad_sequences([[1, 2, 3], [4, 5]])
    → tensor([[1, 2, 3],
              [4, 5, 0]])
    """
    max_len = max(len(s) for s in sequences)
    # Right-pad each sequence with pad_idx to reach max_len
    padded = [s + [pad_idx] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long)


# Encode the whole corpus, then pad into a single matrix
encoded_corpus = [vocab.encode(tok) for tok in tokenized_corpus]
padded_batch = pad_sequences(encoded_corpus)  # (6, 6)

print(f"\nPadded batch shape : {padded_batch.shape}")
print(padded_batch)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

EMBED_DIM = 16   # dimensionality of token embeddings


def build_embedding_layer(vocab_size: int,
                           embed_dim: int,
                           pad_idx: int = 0) -> nn.Embedding:
    """
    Create an nn.Embedding table with a frozen zero vector for <PAD>.

    Setting padding_idx=pad_idx does two things:
      1. Initialises the <PAD> row to all zeros.
      2. Zeroes out the gradient for that row after every backward pass,
         so the <PAD> vector is never updated during training.

    Parameters
    ----------
    vocab_size : number of tokens in the vocabulary (rows in the table)
    embed_dim  : dimensionality of each embedding vector (columns)
    pad_idx    : index of the padding token (default 0)

    Returns
    -------
    nn.Embedding of shape (vocab_size, embed_dim)
    """
    return nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)


embedding_layer = build_embedding_layer(len(vocab), EMBED_DIM)

print("\n" + "=" * 60)
print("PART 3 — Embeddings")
print("=" * 60)
# <PAD> embedding must be all zeros
pad_emb = embedding_layer(torch.tensor([0]))
print(f"<PAD> embedding all-zero: {pad_emb.abs().sum().item() == 0.0}")

# Embed the whole padded batch
embedded = embedding_layer(padded_batch)   # (B, T, D)
print(f"Embedded batch shape    : {embedded.shape}  (B=6, T=6, D={EMBED_DIM})")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — MLP with Masked Mean Pooling
# ═══════════════════════════════════════════════════════════════════════════════

# ── 4a. Masked mean pooling ───────────────────────────────────────────────────

def masked_mean_pool(embeddings: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Average token embeddings, ignoring positions that correspond to <PAD>.

    A naive mean over all T positions would dilute the representation because
    <PAD> vectors are all-zero and pull the average toward zero.  By masking
    we only average over the real tokens in each sentence.

    Parameters
    ----------
    embeddings     : FloatTensor (B, T, D)
    attention_mask : BoolTensor  (B, T)  — True for real tokens, False for PAD

    Returns
    -------
    FloatTensor (B, D) — one sentence embedding per example in the batch

    How it works
    ------------
    1. Expand the (B, T) mask to (B, T, 1) and multiply with embeddings
       so padding positions become zero vectors.
    2. Sum over T to get (B, D).
    3. Divide each row by the number of real tokens in that row.
    """
    # (B, T) → (B, T, 1) → broadcast to (B, T, D)
    mask_expanded = attention_mask.unsqueeze(-1).float()

    # Zero out padding positions, then sum across the time dimension
    sum_emb = (embeddings * mask_expanded).sum(dim=1)        # (B, D)

    # Number of real (non-padding) tokens per example  →  (B, 1)
    token_counts = attention_mask.sum(dim=1, keepdim=True).float()

    # Divide; clamp prevents division by zero for hypothetical all-PAD rows
    return sum_emb / token_counts.clamp(min=1e-9)            # (B, D)


# ── Quick test ────────────────────────────────────────────────────────────────
attention_mask = (padded_batch != 0)   # True wherever token is not <PAD>
pooled = masked_mean_pool(embedded, attention_mask)
print("\n" + "=" * 60)
print("PART 4 — MLP with Mean Pooling")
print("=" * 60)
print(f"Pooled tensor shape: {pooled.shape}  (B=6, D={EMBED_DIM})")


# ── 4b. MLP classifier ────────────────────────────────────────────────────────

class TextMLP(nn.Module):
    """
    Bag-of-words text classifier.

    Architecture:
      Embedding (frozen PAD)
        → Masked mean pooling      [collapses the T dimension]
        → Linear(embed_dim, hidden_dim)
        → ReLU
        → Dropout
        → Linear(hidden_dim, num_classes)
        → raw logits  (apply nn.CrossEntropyLoss outside)

    Because mean pooling treats all tokens equally and discards word order,
    this model works best when the task relies on the *presence* of key words
    rather than their sequence (e.g. keyword-based sentiment, topic detection).
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 pad_idx: int = 0,
                 dropout: float = 0.3):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=pad_idx)
        self.fc1       = nn.Linear(embed_dim, hidden_dim)
        self.relu      = nn.ReLU()
        self.dropout   = nn.Dropout(dropout)
        self.fc2       = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : Tensor (B, T) — padded token indices

        Returns
        -------
        FloatTensor (B, num_classes) — raw (pre-softmax) class scores
        """
        # 1. Token index → dense vector   (B, T) → (B, T, D)
        emb = self.embedding(input_ids)

        # 2. Build attention mask: True where the token is not <PAD>  (B, T)
        mask = (input_ids != self.pad_idx)

        # 3. Collapse the T dimension via masked mean   (B, T, D) → (B, D)
        pooled = masked_mean_pool(emb, mask)

        # 4. Two-layer MLP with ReLU and dropout
        x = self.fc1(pooled)   # (B, D) → (B, H)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)        # (B, H) → (B, num_classes)
        return x


mlp = TextMLP(vocab_size=len(vocab), embed_dim=EMBED_DIM,
              hidden_dim=32, num_classes=2)
mlp_logits = mlp(padded_batch)
print(f"TextMLP output shape   : {mlp_logits.shape}  (B=6, num_classes=2)")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — RNN Classifier
# ═══════════════════════════════════════════════════════════════════════════════

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
                 num_classes: int,
                 pad_idx: int = 0):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=pad_idx)
        # batch_first=True  →  input/output tensors are (B, T, *)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
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
        # 1. Token index → dense vector   (B, T) → (B, T, D)
        emb = self.embedding(input_ids)

        # 2. Pack: merge variable-length sequences into a compact format.
        #    enforce_sorted=False lets PyTorch handle the internal length-sort
        #    so we don't have to pre-sort the batch ourselves.
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True,
                                      enforce_sorted=False)

        # 3. Run the RNN.
        #    _output  : packed representation of all hidden states (not used here)
        #    hidden   : final hidden state, shape (num_layers, B, H) = (1, B, H)
        _output, hidden = self.rnn(packed)

        # 4. (Optional) unpack output if you need all hidden states:
        #    output, _ = pad_packed_sequence(_output, batch_first=True)
        #    — not needed when using only the final hidden state.

        # 5. Drop the num_layers dimension → (B, H)
        hidden = hidden.squeeze(0)

        # 6. Linear classifier  (B, H) → (B, num_classes)
        return self.fc(hidden)


# Compute real sequence lengths (count of non-PAD tokens per row)
lengths = (padded_batch != 0).sum(dim=1)   # LongTensor (B,)

rnn = TextRNN(vocab_size=len(vocab), embed_dim=EMBED_DIM,
              hidden_dim=32, num_classes=2)
rnn_logits = rnn(padded_batch, lengths)

print("\n" + "=" * 60)
print("PART 5 — RNN Classifier")
print("=" * 60)
print(f"Sequence lengths       : {lengths.tolist()}")
print(f"TextRNN output shape   : {rnn_logits.shape}  (B=6, num_classes=2)")


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop — end-to-end demonstration
# ═══════════════════════════════════════════════════════════════════════════════

def train(model, padded_batch, lengths, targets, epochs=60, lr=1e-3, use_lengths=False, log_interval = 10):
    """
    Minimal training loop for demonstration purposes.

    Uses Adam + CrossEntropyLoss on the full batch (no train/val split).
    Reports loss and accuracy every log_interval epochs.

    Parameters
    ----------
    model        : TextMLP or TextRNN
    padded_batch : LongTensor (B, T)
    lengths      : LongTensor (B,) — only used by TextRNN
    targets      : LongTensor (B,) — class labels
    use_lengths  : set True for TextRNN (passes lengths to forward)
    """
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimiser.zero_grad()

        # Forward pass
        logits = model(padded_batch, lengths) if use_lengths else model(padded_batch)

        loss = criterion(logits, targets)
        loss.backward()
        optimiser.step()

        if epoch % log_interval == 0:
            preds = logits.argmax(dim=1)
            acc   = (preds == targets).float().mean().item()
            print(f"  Epoch {epoch:3d} | loss {loss.item():.4f} | acc {acc:.2f}")


targets = torch.tensor(labels)

print("\n--- Training TextMLP ---")
mlp_model = TextMLP(len(vocab), EMBED_DIM, hidden_dim=32, num_classes=2)
train(mlp_model, padded_batch, lengths, targets, use_lengths=False, epochs=10, log_interval=1)

print("\n--- Training TextRNN ---")
rnn_model = TextRNN(len(vocab), EMBED_DIM, hidden_dim=32, num_classes=2)
train(rnn_model, padded_batch, lengths, targets, use_lengths=True, epochs=10, log_interval=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Inference helper
# ═══════════════════════════════════════════════════════════════════════════════

def predict(model, sentence: str, vocab: Vocabulary,
            use_lengths: bool = False) -> str:
    """
    Run a single sentence through a trained model and return the predicted label.

    Steps:
      1. Tokenise and encode the sentence.
      2. Add a batch dimension (B=1).
      3. Forward pass → argmax → human-readable label.
    """
    model.eval()
    with torch.no_grad():
        tokens  = whitespace_tokenize(sentence)
        ids     = torch.tensor([vocab.encode(tokens)])   # (1, T)
        lengths = torch.tensor([(ids != 0).sum().item()])

        logits  = model(ids, lengths) if use_lengths else model(ids)
        pred    = logits.argmax(dim=1).item()

    return "positive" if pred == 1 else "negative"


print("\n" + "=" * 60)
print("Inference examples")
print("=" * 60)
for sent in ["this movie was absolutely brilliant",
             "the worst film i have ever seen"]:
    mlp_pred = predict(mlp_model, sent, vocab, use_lengths=False)
    rnn_pred = predict(rnn_model, sent, vocab, use_lengths=True)
    print(f"  \"{sent}\"")
    print(f"    MLP → {mlp_pred}  |  RNN → {rnn_pred}")
