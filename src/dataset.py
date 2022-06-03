import torch
import torch.utils.data as Data
from transformers import BertTokenizer, BertModel, AdamW, AutoTokenizer, AutoModel
import transformers
import os
import copy
import pdb
import csv
import random
import json

def read_csv_file(path):
    samples = []
    with open(path, encoding='utf-8-sig')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            samples.append(row)
    return samples

def read_txt_file(path):
    samples = []
    with open(path)as f:
        lines = f.readlines()
        for row in lines:
            samples.append(row.strip())
    return samples

class BertDataset(Data.Dataset):
    def __init__(self, args, sample_path, recall_file_path, neg_rate=10): # csv origin file
        # pdb.set_trace()
        self.args = args
        self.sample_path = sample_path
        self.neg_rate = neg_rate
        self.samples = read_csv_file(self.sample_path) # all samples
        self.golds = [x[0] for x in self.samples]
        self.golds_set = list(set(self.golds))
        self.golds_pos = [self.golds_set.index(x) for x in self.golds]
        self.riddles = [x[1] for x in self.samples]
        # chaizi
        self.chaizi_path = args.chaizi_file
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
                val = val.replace(" ", "")
                self.chaizi_dict[key] = val
        # chai golds
        self.golds_radicle = [self.chaizi_dict[x] if x in self.chaizi_dict else "" for x in self.golds] # N len
        self.golds_set_radicle = [self.chaizi_dict[x] if x in self.chaizi_dict else "" for x in self.golds_set] # G len
        # chai riddle
        if (args.use_riddle_radicle):
            def to_radicle(s):
                ret = ""
                for c in s:
                    ret += self.chaizi_dict[c] if c in self.chaizi_dict else ""
                return ret
            self.riddles_radicle = [to_radicle(s) for s in self.riddles]
        # use recall
        if (args.use_recall):
            # pdb.set_trace()
            self.recall_data = read_txt_file(recall_file_path)
            if (len(self.recall_data) != len(self.riddles)):
                print("************error: recall data len %d != riddle len %d****************"%(len(self.recall_data), len(self.riddles)))
                args.use_recall = False
            self.recall_list = [[char for char in line] for line in self.recall_data] # N len [[],[],...]
            self.recall_list_gold_pos = [self.recall_list[i].index(self.golds[i]) \
                                                        if self.golds[i] in self.recall_list[i] else -1 \
                                                                for i in range(len(self.recall_data))]

    def __getitem__(self, index):
        riddle = ""
        ans = ""
        label = 0
        riddle = self.riddles[index // self.neg_rate]
        if (self.args.use_riddle_radicle):
            riddle = self.riddles[index // self.neg_rate] + self.riddles_radicle[index // self.neg_rate]
        if index % self.neg_rate == 0:
            ans = self.golds_radicle[index // self.neg_rate]
            label = 1
        else:
            if (self.args.use_recall):
                riddle_recall = self.recall_list[index // self.neg_rate]
                ans_char = riddle_recall[random.randint(0, len(riddle_recall)-1)]
                ans = self.chaizi_dict[ans_char] if ans_char in self.chaizi_dict else ""
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
    def save_gold_chaizi_set(self):
        ret = set()
        with open("../data/dict.txt") as f:
            lines = f.readlines()
            for line in lines:
                gold = line[0]
                gold_radicle = self.chaizi_dict[gold] if gold in self.chaizi_dict else ""
                for c in gold_radicle:
                    ret.add(c)
                ret.add(gold)
        # for gold_redicle in self.golds_set_radicle:
        #     for x in gold_redicle:
        #         ret.add(x)
        # for gold in self.golds:
        #     ret.add(gold)

        with open("../gold_chaizi_set_test.json", "w", encoding="utf-8") as f:
            json.dump(list(ret), f, indent=4, ensure_ascii=False)



class BertTestDataset(BertDataset):
    def __init__(self, args, sample_path, recall_file_path): # csv origin file
        super(BertTestDataset, self).__init__(args, sample_path, recall_file_path)

    
    def __getitem__(self, index):
        riddle = self.riddles[index]
        if (self.args.use_riddle_radicle):
            riddle += self.riddles_radicle[index]
        recall = self.golds_set_radicle
        label = self.golds_pos[index]
        if (self.args.use_recall):
            recall = [self.chaizi_dict[x] if x in self.chaizi_dict else "" for x in self.recall_list[index]] # G len
            label = self.recall_list_gold_pos[index]
        
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

class BertPredDataset(Data.Dataset):
    def __init__(self, args): # test file, no label
        # pdb.set_trace()
        self.args = args
        self.sample_path = args.test_file
        self.gold_path = args.gold_file

        self.golds_set = read_txt_file(self.gold_path)
        self.riddles = read_txt_file(self.sample_path)

        # chaizi
        self.chaizi_path = args.chaizi_file
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
                val = val.replace(" ", "")
                self.chaizi_dict[key] = val
        # chai golds
        self.golds_set_radicle = [self.chaizi_dict[x] if x in self.chaizi_dict else "" for x in self.golds_set] # G len
        # chai riddle
        if (args.use_riddle_radicle):
            def to_radicle(s):
                ret = ""
                for c in s:
                    ret += self.chaizi_dict[c] if c in self.chaizi_dict else ""
                return ret
            self.riddles_radicle = [to_radicle(s) for s in self.riddles]
        # use recall
        if (args.use_recall):
            # pdb.set_trace()
            self.recall_data = read_txt_file(self.args.test_recall_file)
            if (len(self.recall_data) != len(self.riddles)):
                print("************error: recall data len %d != riddle len %d****************"%(len(self.recall_data), len(self.riddles)))
                args.use_recall = False
            self.recall_list = [[char for char in line] for line in self.recall_data] # N len [[],[],...]
            

    def __getitem__(self, index):
        riddle = self.riddles[index]
        if (self.args.use_riddle_radicle):
            riddle += self.riddles_radicle[index]
        recall = self.golds_set_radicle
        if (self.args.use_recall):
            recall = [self.chaizi_dict[x] if x in self.chaizi_dict else "" for x in self.recall_list[index]] # G len
        return (riddle, recall)

    def __len__(self):
        return len(self.riddles)

if __name__ == "__main__":
    dataset = BertDataset(None, "../data/train.csv", "../data/chaizi-jt.txt", "bert-base-chinese", neg_rate=10)
    dataset.save_gold_chaizi_set()