import dataloader as dl
import model_1 as m1
import model_2 as m2
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch

def plot(T1, T2):
    # TODO
    plt.show()

if __name__ == "__main__":
    train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = dl.load_data()
    vocab = dl.Vocabulary()
    vocab.build_vocabulary(train_inputs)
    encoded_train_corpus = vocab.encode(train_inputs)
    train_targets = torch.tensor(train_labels)
    train_lengths = (encoded_train_corpus != 0).sum(dim=1)   # LongTensor (B × 1)

    encoded_val     = vocab.encode(val_inputs)
    val_lengths = (encoded_val != 0).sum(dim=1)
    val_targets = torch.tensor(val_labels)

    EMBED_DIM = 256
    HIDDEN_DIM = 64
    NUM_LAYERS = 1
    LEARNING_RATE = 1e-2
    EPOCHS = 100

    model_1 = m1.get_model(vocab, EMBED_DIM, HIDDEN_DIM,NUM_LAYERS)
    print(f"\n--- Training TextRNN with lr: {LEARNING_RATE} on {EPOCHS} epocs ---")
    T1 = m1.train(model_1, encoded_train_corpus, train_lengths, train_targets,
                  val_ids=encoded_val, val_lengths=val_lengths, val_targets=val_targets,
                  use_lengths=True, epochs=EPOCHS, lr=LEARNING_RATE, log_interval=1)

    # model_2 = m1.get_model(vocab, EMBED_DIM, HIDDEN_DIM,NUM_LAYERS)
    # print(f"\n--- Training TextRNN with lr: {LEARNING_RATE} on {EPOCHS} epocs ---")
    # T2 = m1.train(model_1, encoded_train_corpus, lengths, targets, use_lengths=True, epochs=EPOCHS, lr=LEARNING_RATE, log_interval=1)

    #plot(T1, T2)

    print(f"\n--- Testing TextRNN ---")
    for sent in ["i feel so happy today",
                 "i am terrified of what comes next",
                 "Nikolaj is the ugliest most stupid and annoying person to walk this earth"]:
        rnn_pred = m1.predict(model_1, sent, vocab, use_lengths=False)
        print(f"  \"{sent}\" → {rnn_pred}")













