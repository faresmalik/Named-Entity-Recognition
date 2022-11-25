import torch 
from tqdm import tqdm 
import numpy as np 

class Trainer_Evaluater(): 
    def __init__(self, model, num_epochs,optimizer,tag_map, vocab, device, criterion, print_every = 5) -> None:
        self.model  = model
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.tag_map = tag_map
        self.print_every = print_every
        self.device = device
        self.criterion = criterion
        self.vocab = vocab

    def fit(self, train_loader):
        self.model.train()
        trainin_loss = []
        trainin_acc = [] 
        for epoch in range(self.num_epochs): 
            print(f'Epoch {epoch+1} / {self.num_epochs}')
            running_loss = 0.0 
            acc = 0.0 
            for a in tqdm(train_loader): 
                self.optimizer.zero_grad()
                sample, labels, mask = a[0].to(self.device), a[1].to(self.device), a[-1].to(self.device)
                outputs = self.model(sample)
                loss = self.criterion(outputs.transpose(1,2),labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() 
                predictions = torch.argmax(outputs, dim=-1)
                mask = labels != self.tag_map['<PAD>']
                a = predictions[mask]
                b = labels[mask]
                accuracy = torch.sum(a == b).item() / a.shape[0]
                acc += accuracy
            #append training loss and acc
            trainin_loss.append(running_loss/len(train_loader))
            trainin_acc.append(acc/len(train_loader))

            if (epoch+1)%self.print_every == 0 and epoch!= 0:
                print('=======================')
                print(f'Loss = {(running_loss/len(train_loader)):.3f}')
                print(f'Acc = {(acc/len(train_loader))*100:.3f}%')
                print('=======================')
        
        return trainin_acc, trainin_loss
    
    def evaluate(self, test_loader):
        self.model.eval()
        acc = 0.0 
        for a in tqdm(test_loader): 
            with torch.no_grad():
                sample, labels, mask = a[0].to(self.device), a[1].to(self.device), a[-1].to(self.device)
                outputs = self.model(sample)
                predictions = torch.argmax(outputs, dim=-1)
                mask = labels != self.tag_map['<PAD>']
                a = predictions[mask]
                b = labels[mask]
                accuracy = torch.sum(a == b).item() / a.shape[0]
                acc += accuracy
        print(f'Evaluation Acc = {(acc/len(test_loader))*100:.3f}%')
        return acc

    def predict(self, sentence):
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
        self.model.eval()
        s = [self.vocab[token] if token in self.vocab else self.vocab['UNK'] for token in sentence.split(' ')]
        batch_data = np.ones((1, len(s)))
        batch_data[0][:] = s
        s = np.array(batch_data).astype(int)
        output = self.model(torch.tensor(s).cuda())
        outputs = torch.argmax(output, axis=2)
        labels = list(self.tag_map.keys())
        pred = []
        for i in range(len(outputs[0])):
            idx = outputs[0][i] 
            pred_label = labels[idx]
            pred.append(pred_label)
        
        for x,y in zip(sentence.split(' '), pred):
            if y != 'O':
                print(x,'\t', y)
        return pred