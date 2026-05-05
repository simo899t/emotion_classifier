from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    print(f"Split representation in pct%: Train: {freq_train_labels}, Validation: {freq_val_labels}, Test: {freq_test_labels}")
    print(f"Accuracy of a dominance classifier on each split: Train: {dom_classifier_train}, Validation: {dom_classifier_val}, Test: {dom_classifier_test}")

    if show_plot:
        emotions = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
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





#step 2)

#Convert input text to a list of tokens for each sentence 
def whitespace_tokenize(text): #Splittin into tokenzzzz
    raw_text = [str(s).split() for s in text] # split and lower case
    
    # res = raw_text.lower().split() #Tokenization
    #no_dup = list(dict.fromkeys(res)) #Remove duplicates 
    return  raw_text

# Analyse the length of texts (characters)
def analyse_text_lengths_chars(tokenized_input):
    text_lengths = np.array([sum(len(token) for token in text) for text in tokenized_input])
    length = len(text_lengths)
    mean = np.sum(text_lengths) / length
    variance = np.sum(np.square(text_lengths-mean))/length
    # variance = sum((x - mean) ** 2 for x in text_lengths) / length
    std_deviation = np.sqrt(variance)
    max, min = np.max(text_lengths), np.min(text_lengths)

    return mean, variance, std_deviation, max, min

# Analyse the length of texts (tokens)
def analyse_text_lengths_words(tokenized_input):
    text_lengths = np.array([len(text) for text in tokenized_input])
    length = len(text_lengths)
    mean = np.sum(text_lengths) / length
    variance = np.sum(np.square(text_lengths-mean))/length
    # variance = sum((x - mean) ** 2 for x in text_lengths) / length
    std_deviation = np.sqrt(variance)
    
    range = (np.min(text_lengths), np.max(text_lengths)) # (min, max)
    return mean, variance, std_deviation, range

# Plot the length of train, validation and test texts (characters)
def plot_text_lengths_chars(tokenized_train_inputs, tokenized_val_inputs, tokenized_test_inputs):
    df = pd.DataFrame(
        [(sum(len(token) for token in t), "Train") for t in tokenized_train_inputs]
        + [(sum(len(token) for token in t), "Validation") for t in tokenized_val_inputs]
        + [(sum(len(token) for token in t), "Test") for t in tokenized_test_inputs],
        columns=["num_chars", "split"],
    )
    g = sns.displot(
        df, x="num_chars", col="split",
        binwidth=1, height=3, facet_kws=dict(margin_titles=True),
    )
    g.set_axis_labels("num_chars", "num_sentences")
    plt.show()

# Plot the length of train, validation and test texts (tokens)
def plot_text_lengths_words(tokenized_train_inputs, tokenized_val_inputs, tokenized_test_inputs):
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


if __name__ == '__main__':
    train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = load_data()
    #analyze_class_distribution(train_labels, val_labels, test_labels)
    tokenized_train_input, tokenized_validation_input, tokenized_test_input = whitespace_tokenize(train_inputs), whitespace_tokenize(val_inputs), whitespace_tokenize(test_inputs)
    
    mean, variance, std, max, min = analyse_text_lengths_words(tokenized_train_input)
    print(mean, variance, std, max, min)

    plot_text_lengths_words(tokenized_train_input, tokenized_validation_input, tokenized_test_input)



