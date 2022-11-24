import torch.nn as nn 


class NER_model(nn.Module): 
    def __init__(self, vocab, tag_map, d_model, num_layers = 2, max_length = 100, batch_first = True) -> None:
        super(NER_model, self).__init__()

        self.hidden_dim = d_model
        self.vocab_size = len(vocab)
        self.num_layers = num_layers 
        self.num_classes = len(tag_map)
        self.max_length = max_length

        self.Embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.hidden_dim)
        self.LSTM = nn.LSTM(input_size = self.hidden_dim, hidden_size = self.hidden_dim, num_layers= self.num_layers, batch_first= batch_first)
        self.FC = nn.Linear(in_features= self.hidden_dim, out_features=self.num_classes)
        
    def forward(self, x): 
        x = self.Embedding(x)
        x,_ = self.LSTM(x)
        x = self.FC(x)
        return x