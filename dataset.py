import torch 
from torch.utils.data import Dataset, DataLoader

class Data_class(Dataset):
    """
    Dataset class to creat dataset. 

    Args: 
        vocab (dict)
        tag_map (dict)
        data (list)
        labels (list)
        max_length (int)
        padding (str)
    
    Return: 
        sample and its label 

    """
    def __init__(self, vocab, tag_map, data, labels, max_length ,padding = '<PAD>'):
        
        self.data = data 
        self.labels = labels
        self.tag_map = tag_map
        self.padding = padding
        self.sentences_list, self.labels_list = self.sent_to_tensor(vocab, tag_map, data, labels)
        
        # Load the vocabulary mappings
        self.words_to_idx = vocab
        self.idx_to_words = {str(idx): word for word, idx in vocab.items()}

        self._dataset_size = len(self.data)

        self._max_len = max_length

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        sample = self.sentences_list[index]
        sample_size = len(sample)
        label = self.labels_list[index]
        
        # Pad the token and label sequences
        sample = sample[:self._max_len]
        labels = label[:self._max_len]
        padding_size = self._max_len - sample_size

        if padding_size > 0:
            sample += [self.words_to_idx[self.padding] for _ in range(padding_size)]
            labels += [self.tag_map[self.padding] for _ in range(padding_size)]

        sample = torch.Tensor(sample).long()

        # Adapt labels for PyTorch consumption
        labels = [int(label) for label in labels]
        labels = torch.Tensor(labels).long()

        # Define the padding mask
        padding_mask = torch.ones([self._max_len, ])
        padding_mask[:sample_size] = 0.0
        return sample, labels, padding_mask.long()

    def sent_to_tensor(self, vocab, tag_map, sentences, labels):
        sentences_list = []
        labels_list = []
        for sentence in sentences:
            sent_inter = []
            for word in sentence.split(' '):
                if word in vocab: 
                    sent_inter.append(vocab[word])
                elif word not in vocab: 
                    sent_inter.append(vocab['UNK'])
            sentences_list.append(sent_inter)

        for label in labels: 
            tag_inter = []
            for tag in label.split(' '): 
                tag_inter.append(tag_map[tag])
            labels_list.append(tag_inter)

        return sentences_list, labels_list