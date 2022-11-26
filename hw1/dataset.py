from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab
from torch.nn.utils.rnn import pad_sequence

import torch
import numpy as np
import torch.nn.functional as F



from torch.utils.data import Dataset
import torch
import numpy as np
from utils import Vocab

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    #對樣本進行處理
    def collate_fn(self, samples: List[Dict]) -> Dict:
        #對text,label處理
        output = {}
        text = []
        intentOutput = []
        ids = []
        for i in range(len(samples)):
            text.append(samples[i]["text"].split(' '))
        
        textOutput = Vocab.encode_batch(self.vocab,text)

        if len(samples[0]) == 3:
            for i in range(len(samples)):
                intentOutput.append(self.label2idx(samples[i]["intent"]))
                output["intent"] = torch.tensor(intentOutput,dtype=torch.long)
        
        for i in range(len(samples)):
            ids.append(samples[i]["id"])
        output["id"] = ids
        output["text"] = torch.tensor(textOutput,dtype=torch.long)
        return output
        #intent = torch.tensor(intentOutput,dtype=torch.long)
        #output["intent"] = F.one_hot(intent).float()
        #print(output["intent"])
        #print(output["text"].shape)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class SeqTaggingClsDataset(Dataset):
    ignore_idx = -100
    
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        tagMapping: Dict[str, int],
        max_len: int,
        padValue: int
    ):
        self.data = data
        self.vocab = vocab
        self.tagMapping = tagMapping
        self._idx2tag = {idx: tag for tag, idx in self.tagMapping.items()}
        self.max_len = max_len
        self.padValue = padValue
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def get_num_class(self):
        return len(self.tagMapping)    

    def collate_fn(self, batch):
        #print(batch)
        '''
        #sort
        oriLen = [len(b["tokens"])for b in batch]
        idx = np.array(oriLen).argsort()
        idx = np.flip(idx)
        #print(idx)
        samples = []
        for i in idx:
           samples.append(batch[i])
        print(samples)
        '''
        samples = batch
        output = {}
        #to list(list(str))
        output["tokens"] = [sample["tokens"] for sample in samples]
        output["tokens"] = Vocab.encode_batch(self.vocab,output["tokens"],self.max_len)
        output["tokens"] = torch.tensor(output["tokens"],dtype=torch.long)
        #test file no label
        tag2d = []
        if len(samples[0]) != 2:
            for sample in samples:
                tag1d = []
                for tag in sample["tags"]:
                    tag1d.append(self.tag2idx(tag))
                for i in range(self.max_len - len(sample["tags"])):
                    tag1d.append(self.padValue)
                tag2d.append(tag1d)
        output["tags"] = torch.tensor(tag2d,dtype=int)
        output["originalLen"] = [len(sample["tokens"]) for sample in samples]
        output["originalLen"] = torch.tensor(output["originalLen"],dtype=int)
        output["idx"] = [sample["id"] for sample in samples]
        #print(output)
        #exit(0)
        return output
 
    def tag2idx(self,tag:str):
        return self.tagMapping[tag]

    def idx2tag(self,idx:int):
        return self._idx2tag[idx]
