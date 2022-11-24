import torch 

def sent_to_tensor(vocab, tag_map, sentences, labels):
    """
    Take sentences with labels, and return lists of integers. 

    Args: 
        vocab (dict)
        tag_map (dict) 
        sentences (list)
        labels (list)

    Return: 
        sentences_list (list)
        labels_list (list)
    """

    sentences_list = []
    labels_list = []
    for sentence in sentences:
        sent_inter = []
        for word in sentence.split(' '):
            if word in vocab: 
                sent_inter.append(vocab[word])
            elif word not in vocab: 
                sent_inter.append(vocab['UNK'])
        sentences_list.append(torch.tensor(sent_inter))

    for label in labels: 
        tag_inter = []
        for tag in label.split(' '): 
            tag_inter.append(tag_map[tag])
        labels_list.append(torch.tensor(tag_inter))

    return sentences_list, labels_list


def read_txt_file(text_path, mode = 'r'): 
    """
    Read text file and return a list. 

    Args: 
        text_path (str): directory of the text file. 
    
    Return:
        list of lines without the \n 

    """

    f = open(text_path, mode)
    list_sent = f.readlines()
    list_sent = [line.rstrip('\n') for line in list_sent]
    return list_sent

def get_vocab_tag (vocab_sent, vocab_tags, padding = '<PAD>'):
    """
    Get the vocabulary dict and tags dict

    Args: 
        vocab_sent (list): list of sentences to obtain the vocabulary 
        vocab_tags (list): list of possible tags
        padding (str): padding token
    
    Return: 
        vocab (dict): dict that maps words to int 
        tag_map (dict): dict that maps tag to int
    
    """
    vocab = {}
    tag_map  = {}
    i= 0
    j= 0
    for word in vocab_sent:
        if word not in vocab: 
            vocab[word] = i
            i+=1 
    vocab[padding] = len(vocab)

    for tag in vocab_tags: 
        if tag not in tag_map: 
            tag_map[tag] = j
            j+=1 
    tag_map[padding] = len(tag_map)
    return vocab, tag_map