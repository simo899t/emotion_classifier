

def whitespace_tokenize(text):
    raw_text = ' '.join([str(s) for s in text]) #Remove the list and joining into one big string
    res = raw_text.lower().split() #Tokenization
    no_dup = list(dict.fromkeys(res)) #Remove duplicates 
    return  no_dup



te_text = ["Hey how how are you you"]

clear_text = whitespace_tokenize(te_text)
print(clear_text)