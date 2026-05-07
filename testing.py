import dataloader as dl
import rnn
import lstm
import transformer as tf
import pandas as pd
import matplotlib.pyplot as plt
import torch

def plot(T1, T2):
    f, (ax1, ax2) = plt.subplots(1, 2) 
    ax1.plot(T1[0], T1[1], label='T1')
    ax1.plot(T1[0], T2[1], label='T2')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(T2[0], T1[2], label='T1')
    ax2.plot(T2[0], T2[2], label='T2')
    ax2.set_title('Acc')
    ax2.legend()

    plt.tight_layout()
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

    EMBED_DIM = 64
    HIDDEN_DIM = 64
    NUM_LAYERS = 1
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    BATCH_SIZE = 32
    D_MODEL = 64
    D_KEYS = 32
    N_HEAD = 4
    MLP_FACTOR = 2

    # rnn_model = rnn.get_model(vocab, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS)
    # print(f"\n--- Training TextRNN with lr: {LEARNING_RATE} on {EPOCHS} epocs ---")
    # T1 = rnn.train(rnn_model, encoded_train_corpus, train_lengths, train_targets,
    #               val_ids=encoded_val, val_lengths=val_lengths, val_targets=val_targets,
    #               use_lengths=True, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, log_interval=1)
    #
    # lstm_model = lstm.get_model(vocab, EMBED_DIM, HIDDEN_DIM,NUM_LAYERS)
    # print(f"\n--- Training TextRNN with lr: {LEARNING_RATE} on {EPOCHS} epocs ---")
    # T2 = lstm.train(lstm_model, encoded_train_corpus, train_lengths, train_targets,
    #               val_ids=encoded_val, val_lengths=val_lengths, val_targets=val_targets,
    #               use_lengths=True, epochs=EPOCHS, lr=LEARNING_RATE, log_interval=1)
    
    tf_model = tf.get_model(vocab, D_MODEL, D_KEYS, N_HEAD, MLP_FACTOR, NUM_LAYERS)
    print(f"\n--- Training TextRNN with lr: {LEARNING_RATE} on {EPOCHS} epocs ---")
    T3 = tf.train(tf_model, encoded_train_corpus, train_targets,
              val_ids=encoded_val, val_targets=val_targets,
              epochs=EPOCHS, lr=LEARNING_RATE, log_interval=1)



    # plot(T2, T3)

    order_test_pairs = [
    # Pair A: same emotion words, opposite endings
    "i felt joy then i felt anger",      # → anger expected
    "i felt anger then i felt joy",      # → joy expected

    # Pair B: feeling shifted to opposite
    "i was happy but now i feel sad",    # → sadness expected
    "i was sad but now i feel happy",    # → joy expected

    # Pair C: emotion at start vs end
    "scared at first then surprised",    # → surprise expected
    "surprised at first then scared",    # → fear expected

    # Pair D: simple ordering
    "today was full of fear and joy",    # ambiguous
    "today was full of joy and fear",    # ambiguous
    # both should give DIFFERENT answers if order matters
    ]
    test = ["i feel so happy today",
                 "i am terrified of what comes next",
                 "Nikolaj is the ugliest most stupid and annoying person to walk this earth"]

    print(f"\n--- Testing TextRNN ---")
    for sent in order_test_pairs:
        rnn_pred = rnn.predict(rnn_model, sent, vocab, use_lengths=False)
        print(f"  \"{sent}\" → {rnn_pred}")













