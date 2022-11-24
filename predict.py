import torch
import numpy as np 

def predict(sentence, model, vocab, tag_map):
    """
    Predict the tag for words in a sentence 

    Args: 
        sentence (str): sentence  
        model : trained NER model 
        vocab (dict): vocabulary dict, maps each word to int
        tag_map (dict): dict, maps each tag to int
    Return: 
        pred (list): tag for each word in the input sentence 
    
    Print: 
        The tag other than 'O' tag. 
    
    """
    s = [vocab[token] if token in vocab else vocab['UNK'] for token in sentence.split(' ')]
    batch_data = np.ones((1, len(s)))
    batch_data[0][:] = s
    s = np.array(batch_data).astype(int)
    output = model(torch.tensor(s).cuda())
    outputs = torch.argmax(output, axis=2)
    labels = list(tag_map.keys())
    pred = []
    for i in range(len(outputs[0])):
        idx = outputs[0][i] 
        pred_label = labels[idx]
        pred.append(pred_label)
    
    for x,y in zip(sentence.split(' '), pred):
        if y != 'O':
            print(x,y)
    return pred