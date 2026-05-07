from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns
from matplotlib import colormaps
sns.set_theme(style="darkgrid")





def load_data():
    ds = load_dataset("dair-ai/emotion", "split")
    train_inputs = ds["train"][:]["text"]
    val_inputs = ds["validation"][:]["text"]
    test_inputs = ds["test"][:]["text"]
    train_labels = ds["train"][:]["label"]
    validation_labels = ds["validation"][:]["label"]
    test_labels = ds["test"][:]["label"]    
    return train_inputs, val_inputs, test_inputs, train_labels, validation_labels, test_labels
    # sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)

def check_data_alignment(train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels):
    assert len(train_inputs) == len(train_labels), f"train: {len(train_inputs)} vs {len(train_labels)}"
    assert len(val_inputs) == len(val_labels), f"val: {len(val_inputs)} vs {len(val_labels)}"
    assert len(test_inputs) == len(test_labels), f"test: {len(test_inputs)} vs {len(test_labels)}"
    for name, xs in [("train_inputs", train_inputs), ("val_inputs", val_inputs), ("test_inputs", test_inputs),
                     ("train_labels", train_labels), ("val_labels", val_labels), ("test_labels", test_labels)]:
        assert not any(x is None for x in xs), f"{name} contains None"
    print(f"OK — train: {len(train_inputs)}, val: {len(val_inputs)}, test: {len(test_inputs)}")

# step 1
def analyze_class_distribution(train_labels, val_labels, test_labels, show_plot = True):
    freq_train_labels = np.bincount(np.array(train_labels))
    freq_val_labels = np.bincount(np.array(val_labels))
    freq_test_labels = np.bincount(np.array(test_labels))

    total_train = np.sum(freq_train_labels)
    total_val = np.sum(freq_val_labels)
    total_test = np.sum(freq_test_labels)
    acc_list_train, acc_list_val, acc_list_test = [], [], []
    for train_acc in freq_train_labels: acc_list_train.append(round(float(train_acc/total_train * 100),1))
    for val_acc in freq_val_labels: acc_list_val.append(round(float(val_acc/total_val * 100),1))
    for test_acc in freq_test_labels: acc_list_test.append(round(float(test_acc/total_test * 100),1))

    dom_classifier_train = max(freq_train_labels)/total_train
    dom_classifier_val = max(freq_val_labels)/total_val
    dom_classifier_test = max(freq_test_labels)/total_test

    print(f"Split representation: Train: {total_train}, Validation: {total_val}, Test: {total_test}")
    print(f"Split representation in pct%: Train: {freq_train_labels/total_train}, Validation: {freq_val_labels/total_val}, Test: {freq_test_labels/total_test}")
    print(f"Accuracy of a dominance classifier on each split: Train: {dom_classifier_train}, Validation: {dom_classifier_val}, Test: {dom_classifier_test}")

    if show_plot:
        emotions = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
        # cmap = plt.get_cmap('plasma')
        # colors = cmap(np.linspace(0,1,6))
        colors = ['royalblue', 'gold', 'hotpink', 'firebrick', 'purple', 'orange']
        splits = [
            ("Train", freq_train_labels),
            ("Validation", freq_val_labels),
            ("Test", freq_test_labels),
        ]

        _, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (title, freq) in zip(axes, splits):
            ax.pie(freq, labels=emotions, autopct='%1.1f%%', colors=colors)
            ax.set_title(f"{title} emotion split")
        plt.show()

# max length of text (words)
def _max(tokenized_input):
    text_lengths = np.array([len(text) for text in tokenized_input])
    max = np.max(text_lengths)
    return max

# Plot the length of train, validation and test texts (tokens)
def plot_text_lengths_words(train_inputs, val_inputs, test_inputs):
    tokenized_train_inputs = whitespace_tokenize(train_inputs)
    tokenized_val_inputs = whitespace_tokenize(val_inputs)
    tokenized_test_inputs = whitespace_tokenize(test_inputs)

    df = pd.DataFrame(
        [(len(t), "Train") for t in tokenized_train_inputs]
        + [(len(t), "Validation") for t in tokenized_val_inputs]
        + [(len(t), "Test") for t in tokenized_test_inputs],
        columns=["num_words", "split"],
    )
    g = sns.displot(
        df, x="num_words", col="split",
        binwidth=1, height=3, facet_kws=dict(margin_titles=True),
    )
    g.set_axis_labels("num_words", "num_sentences")
    plt.show()

def analyse_text_lengths_words(raw_input: list[str]):
    tokenized_input = whitespace_tokenize(raw_input)
    text_lengths = np.array([len(text) for text in tokenized_input])
    length = len(text_lengths)
    mean = np.sum(text_lengths) / length
    variance = np.sum(np.square(text_lengths - mean)) / length
    std_deviation = np.sqrt(variance)
    length_range = (np.min(text_lengths).item(), np.max(text_lengths).item())
    return mean, variance, std_deviation, length_range

def whitespace_tokenize(raw_text):
        tokenized_text = [str(s).split() for s in raw_text]
        return  tokenized_text

class Vocabulary:
    def __init__(self):
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}

    def build_vocabulary(self,raw_input: list[str]):
        tokenized_input = whitespace_tokenize(raw_input)
        self.token2idx['<PAD>'], self.token2idx['<UNC>'] = 0,1
        self.idx2token[0], self.idx2token[1] = '<PAD>', '<UNC>'
        idx = 2
        for text in tokenized_input:
            for token in text:
                if token not in self.token2idx:
                    self.token2idx[token] = idx
                    self.idx2token[idx] = token
                    idx += 1

    def encode(self, input):
        tokenized_input = whitespace_tokenize(input)
        max_len = _max(tokenized_input)
        num_sent = len(tokenized_input)
        pad_idx = 0
        x = torch.full((num_sent, max_len), pad_idx)
        for i, sent in enumerate(tokenized_input):
            for j, token in enumerate(sent):
                x[i,j] = self.token2idx.get(token, 1)
        return x

    def decode(self, encoded_input):
        x = []
        for sent in encoded_input:
            x.append([])
            for encoded_token in sent:
                decoded_token = self.idx2token.get(encoded_token.item())
                if decoded_token:
                    x[-1].append(decoded_token)
        return x
        
    def __len__(self) -> int:
        return len(self.token2idx)

if __name__ == '__main__':
    train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = load_data()
    #analyze_class_distribution(train_labels, val_labels, test_labels)
    check_data_alignment(train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels)
    vocab = Vocabulary()
    vocab.build_vocabulary(train_inputs)
    print(vocab.token2idx)
    print(vocab.idx2token)

    #plot_text_lengths_words(train_inputs,val_inputs,test_inputs)

    mean, variance, std, range = analyse_text_lengths_words(test_inputs)
    print(mean, variance, std, range)

    # print(vocab.encode(train_inputs))
    # print(vocab.decode(vocab.encode(train_inputs)))


