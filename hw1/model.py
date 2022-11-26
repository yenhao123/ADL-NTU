from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
       
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.ln = nn.LayerNorm(embeddings.shape[1])
        # output_shape:(batch_size,hiddent_size);let batch_first to true
        self.rnn = nn.GRU(input_size=embeddings.shape[1], 
                        hidden_size=hidden_size, 
                        num_layers=num_layers,
                        bidirectional=bidirectional,
                        dropout=dropout,
                        batch_first=True)
        
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size * (2 if bidirectional else 1))
        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), 512)
        self.act = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_class)

    def forward(self, batch):
        output = self.embed(batch)
        #print(output.shape)
        #output = self.ln(output)
        output,hidden = self.rnn(output)
        #select the last node
        output = output[:,-1,:]
        output = self.bn1(output)
        output = self.act(self.fc1(output))
        output = self.bn2(output)
        output = self.act(self.fc2(output))
        return output,hidden


class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        max_len: int,
        num_class: int,
    ) -> None:
        super(SeqTagger, self).__init__()
        
        self.maxLen = max_len
        #output:(batch_size,str_len,emb_size)
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.ln = nn.LayerNorm(embeddings.shape[1])
        #output:(batch_size,str_len,hidden_size * 2);let batch_first to true
        self.rnn = nn.GRU(input_size=embeddings.shape[1], 
                        hidden_size=hidden_size, 
                        num_layers=num_layers,
                        bidirectional=bidirectional,
                        dropout=dropout,
                        batch_first=True)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_size * (2 if bidirectional else 1))
        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), 512)
        
        #self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), num_class)
        self.act = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_class)
        #self.fc2 = nn.Linear(512, 128)
        #self.bn3 = nn.BatchNorm1d(128)
        #self.fc3 = nn.Linear(128, num_class)
        self.apply(self._init_weights)  
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm1d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()

    def forward(self, data):
        oriLen,batch = data["originalLen"],data["tokens"]
        output = self.embed(batch)
        output = self.ln(output)

        #output = pack_padded_sequence(output, oriLen, batch_first=True, enforce_sorted=False)
        #self.rnn.flatten_parameters()
        output,hidden = self.rnn(output)
        #output,lenUnpack = pad_packed_sequence(output,batch_first=True,padding_value=5.0,total_length=self.maxLen)
        #3 * BN + 3 * FC
        '''
        output = output.permute(0,2,1)
        output = self.bn1(output)
        output = output.permute(0,2,1)
        output = self.act(self.fc1(output))
        output = output.permute(0,2,1)
        output = self.bn2(output)
        output = output.permute(0,2,1)
        output = self.act(self.fc2(output))
        output = output.permute(0,2,1)
        output = self.bn3(output)
        output = output.permute(0,2,1)
        output = self.act(self.fc3(output))
        output = output.permute(0,2,1)
        return output,hidden
        ''' 
        
        #2 * BN + 2 * FC
        
        output = output.permute(0,2,1)
        output = self.bn1(output)
        output = output.permute(0,2,1)
        output = self.act(self.fc1(output))
        output = output.permute(0,2,1)
        output = self.bn2(output)
        output = output.permute(0,2,1)
        output = self.act(self.fc2(output))
        output = output.permute(0,2,1)
        return output,hidden
        

        '''
        #1 * BN + 1 * FC
        output = output.permute(0,2,1)
        output = self.bn1(output)
        output = output.permute(0,2,1)
        output = self.act(self.fc1(output))
        output = output.permute(0,2,1) 
        return output,hidden
        '''
