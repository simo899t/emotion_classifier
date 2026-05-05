import dataloader as dl

train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = dl.load_data()
vocab = dl.Vocabulary()
vocab.build_vocabulary(test_inputs)
encoded_train = vocab.encode(train_inputs)

print("\n" + "=" * 60)
print("PART 2 — Vocabulary")
print("=" * 60)
print(f"Vocabulary size : {len(vocab)}")
print(f"<PAD> index     : {vocab.token2idx['<PAD>']} (should be 0)")
print(f"<UNC> index     : {vocab.token2idx['<UNC>']} (should be 1)")
print(f"Encoded sent[0] : {encoded_train}")
print(f"Decoded back    : {vocab.decode(encoded_train)}")

# Test OOV handling
print(f"Unknown token   : {vocab.encode(['notaword'])} (should be [1])")