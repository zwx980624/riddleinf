import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW, AutoTokenizer, AutoModel

class RadicalEncoder(torch.nn.Module):
    def __init__(self, emb_in, emb_out, num_layers):
        super(RadicalEncoder, self).__init__()
        self.lstm_encoder = nn.LSTM(input_size=emb_size, hidden_size=emb_out//2, \
                                    num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x): # (B, L, Ein)
        output, (hn, cn) = self.lstm_encoder(x)
        return output, (hn, cn) # (B, L, Eout)

class AnsEncoder(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(RadicalEncoder, self).__init__()
        self.n_hidden = torch.nn.Linear(n_feature, n_hidden)
        self.n_hidden2 = torch.nn.Linear(n_hidden, n_hidden // 2)
        self.n_hidden3 = torch.nn.Linear(n_hidden // 2, n_hidden // 4)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, rad, pos): # (B, L, E), (B, L, E)
        x = torch.cat([rad, pos], dim=1).reshape(x.shape[0], -1) # (B, (1+L+L)*E)
        x_layer = torch.relu(self.n_hidden(x))
        x_layer = torch.relu(self.n_hidden2(x_layer))
        x_layer = torch.relu(self.n_hidden3(x_layer))
        x_layer = self.out(x_layer)
        return x_layer # (B, out)

class RiddleModel(torch.nn.Module):
    def __init__(self, args, red_emb_in, red_emb_out, red_num_layers, ans_emb_in, ans_hidden, ans_emb_out):
        super(RiddleModel, self).__init__()
        self.riddle_encoder = BertModel.from_pretrained(args.bert_pretrain_name)
        self.redical_encoder = RedicalEncoder(red_emb_in, red_emb_out, red_num_layers)
        self.ans_encoder = AnsEncoder(ans_emb_in, ans_hidden, ans_emb_out)
 
    def forward(self, riddle, radical, ans):
        riddle_feature = self.riddle_encoder(input_ids=riddle["data"], attention_mask=riddle["mask"])
        redical_feature = self.redical_encoder(x=radical)
        ans_feature = self.ans_encoder(char=ans["char"], rad=ans["rad"], pos=ans["pos"])

        # model

        return x_layer

class Classifier(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Classifier, self).__init__()
        self.n_hidden = torch.nn.Linear(n_feature, n_hidden)
        self.n_hidden2 = torch.nn.Linear(n_hidden, n_hidden // 2)
        self.n_hidden3 = torch.nn.Linear(n_hidden // 2, n_hidden // 4)
        self.out = torch.nn.Linear(n_hidden // 4, n_output)
 
 
    def forward(self, x_layer):
        x_layer = torch.relu(self.n_hidden(x_layer))
        x_layer = torch.relu(self.n_hidden2(x_layer))
        x_layer = torch.relu(self.n_hidden3(x_layer))
        x_layer = self.out(x_layer)
        return x_layer

class RiddleBertModel(torch.nn.Module):
    def __init__(self, args):
        super(RiddleBertModel, self).__init__()
        self.riddle_encoder = BertModel.from_pretrained(args.bert_pretrain_name)
        self.classifiler = Classifier(n_feature=768, n_hidden=1024, n_output=1)
 
    def forward(self, x):
        feature = self.riddle_encoder(input_ids=x["input_ids"], attention_mask=x["attention_mask"], \
                                            token_type_ids=x["token_type_ids"])
        out = self.classifiler(feature["pooler_output"])

        return out