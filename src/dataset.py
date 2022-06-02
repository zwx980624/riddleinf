import torch
import torch.utils.data as Data
from transformers import BertTokenizer, BertModel, AdamW, AutoTokenizer, AutoModel
import transformers
import os
import copy
import pdb
import csv
import random

def read_csv_file(path):
    samples = []
    with open(path, encoding='utf-8-sig')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            samples.append(row)
    return samples

class BertDataset(Data.Dataset):
    def __init__(self, sample_path, chaizi_path, bert_pretrain_name, neg_rate=10): # csv origin file
        # pdb.set_trace()
        self.sample_path = sample_path
        self.neg_rate = neg_rate
        self.samples = read_csv_file(self.sample_path) # all samples
        self.golds = [x[0] for x in self.samples]
        self.golds_set = list(set(self.golds))
        self.golds_pos = [self.golds_set.index(x) for x in self.golds]
        self.riddles = [x[1] for x in self.samples]
        # chaizi
        self.chaizi_path = chaizi_path
        self.chaizi_dict = {}
        with open(self.chaizi_path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\n')[0]
                items = line.split('\t')
                key = items[0]
                val = ""
                for i in range(1, len(items)):
                    val += items[i]
                self.chaizi_dict[key] = val
        # chai golds

        self.golds_radicle = [self.chaizi_dict[x] if x in self.chaizi_dict else "" for x in self.golds] # N len
        self.golds_set_radicle = [self.chaizi_dict[x] if x in self.chaizi_dict else "" for x in self.golds_set] # G len
        

    def __getitem__(self, index):
        riddle = ""
        ans = ""
        label = 0
        riddle = self.riddles[index // self.neg_rate]
        if index % self.neg_rate == 0:
            ans = self.golds_radicle[index // self.neg_rate]
            label = 1
        else:
            rand_idx = random.randint(0, len(self.golds_radicle)-1)
            ans = self.golds_radicle[rand_idx]
        
        # return riddle + "[SEP]" + ans
        return (riddle, ans, label)

    def __len__(self):
        return len(self.riddles) * self.neg_rate

    def my_collate_fn(self, x): # x: [(riddle, ans, label), ... Batch]
        riddle = [_[0] for _ in x]
        ans = [_[1] for _ in x]
        label = [_[2] for _ in x]
        ret = {
            "riddle": riddle,
            "ans": ans,
            "label": label,
        }
        return ret

class BertTestDataset(BertDataset):
    def __init__(self, sample_path, chaizi_path, bert_pretrain_name): # csv origin file
        super(BertTestDataset, self).__init__(sample_path, chaizi_path, bert_pretrain_name)

    
    def __getitem__(self, index):
        riddle = self.riddles[index]
        recall = self.golds_set_radicle
        label = self.golds_pos[index]
        
        return (riddle, recall, label)

    def __len__(self):
        return len(self.riddles)

class RecallDataset(Data.Dataset):
    def __init__(self, recall_list): # csv origin file
        # pdb.set_trace()
        self.recall_list = recall_list
    def __getitem__(self, index):
        return self.recall_list[index]
    def __len__(self):
        return len(self.recall_list)

if __name__ == "__main__":
    dataset = BertDataset("../data/valid.csv", "../data/chaizi-jt.txt", "bert-base-chinese", neg_rate=10)