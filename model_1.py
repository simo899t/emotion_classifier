import dataloader as dl

train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = dl.load_data()

vocab = dl.Vocabulary()
vocab.build_vocabulary(test_inputs)
print(vocab.token2idx)
print(vocab.idx2token)