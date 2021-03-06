import sys
import os
import torch
import logging
import argparse
import torch.utils.data as Data
import torch.nn as nn
import random
import numpy as np
from transformers import BertTokenizer, BertModel, AdamW, AutoTokenizer, AutoModel
from dataset import BertDataset, BertTestDataset, RecallDataset, BertPredDataset
from model import RiddleBertModel
from tqdm import tqdm
import pdb
import copy
from tools import softmax, MRR

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def initial_model(args):

    model = RiddleBertModel(args)
    
    return model


def train_model(args, dataset, val_dataset, model):
    dataloader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, \
                                    collate_fn=dataset.my_collate_fn)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrain_name)

    need_optimized_parameters = []
    for module in [model]:
        need_optimized_parameters += [p for n, p in module.named_parameters() if p.requires_grad]
    optimizer = AdamW([{'params': need_optimized_parameters, 'weight_decay': 0.0}], lr=args.learning_rate)
    loss_func = torch.nn.BCEWithLogitsLoss()

    logger.info("start training")
    best_acc10 = 0.0
    best_mrr = 0.0
    model.train()
    for epoch in range(args.n_epochs):
        logger.info("epoch:{}".format(epoch))
        loop = tqdm(dataloader, total=len(dataloader))
        for idx, data in enumerate(loop):
            riddle = data["riddle"]
            ans = data["ans"]
            label = data["label"]
            label = torch.tensor(label, dtype=torch.float32).reshape([-1,1])
            train_input = tokenizer(text=riddle, text_pair=ans, padding=True, truncation=True, max_length=50, return_tensors='pt')
            if torch.cuda.is_available():
                train_input["input_ids"] = train_input["input_ids"].cuda()
                train_input["token_type_ids"] = train_input["token_type_ids"].cuda()
                train_input["attention_mask"] = train_input["attention_mask"].cuda()
                label = label.cuda()
            logit = model(train_input)
            loss = loss_func(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #loss.detach().cpu().numpy()
            loop.set_postfix(loss = loss.item())
        if (epoch < 3 or epoch == args.n_epochs - 1 or epoch % args.n_val == 0):
            # test
            acc1, acc5, acc10, mrr, rec_fail = test_model(args, val_dataset, model)
            if (mrr > best_mrr):
                # save
                best_mrr = mrr
                logger.info("save model")
                torch.save(model.state_dict(), os.path.join(args.output_dir, "epoch_{}.ckpt".format(epoch)))

def test_model(args, dataset, model):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrain_name)
    # softmax = nn.Softmax(dim=1)
    logger.info("start testing")
    model.eval()
    loop = tqdm(dataset, total=len(dataset))
    label_list = []
    pred_list = []
    acc1 = 0
    acc5 = 0
    acc10 = 0
    recall_failed = 0
    mrr = 0.0
    for idx, data in enumerate(loop):
        riddle = data[0]
        recall_list = data[1]
        label  = data[2]
        if (label == -1):
            recall_failed += 1
            label_list.append(-1)
            pred_list.append([])
            loop.set_postfix(acc10 = acc10/(idx+1))
            continue
        recall_dataset = RecallDataset(recall_list)
        recall_dataloader = Data.DataLoader(dataset=recall_dataset, batch_size=args.batch_size, shuffle=False)
        logit = []
        for recall in recall_dataloader:
            riddle_copy = [riddle for _ in range(len(recall))]
            # pdb.set_trace()
            test_input = tokenizer(text=riddle_copy, text_pair=recall, padding=True, truncation=True, max_length=50, return_tensors='pt')
            if torch.cuda.is_available():
                test_input["input_ids"] = test_input["input_ids"].cuda()
                test_input["token_type_ids"] = test_input["token_type_ids"].cuda()
                test_input["attention_mask"] = test_input["attention_mask"].cuda()
            logit_batch = model(test_input).reshape(-1)
            logit_batch = logit_batch.detach().cpu().numpy().tolist()
            logit += logit_batch
        
        # pdb.set_trace()
        pred = softmax(logit).tolist()
        label_list.append(label)
        pred_list.append(copy.deepcopy(pred))
        # acc
        pred_id = [[pred[i], i] for i in range(len(pred))]
        pred_id.sort(reverse=True, key=lambda x: x[0])
        id_sort = [_[1] for _ in pred_id]
        gold_rank = id_sort.index(label)
        if (gold_rank == 0): acc1 += 1
        if (gold_rank < 5): acc5 += 1
        if (gold_rank < 10): acc10 += 1
        mrr += MRR(gold_rank)
        loop.set_postfix(mrr = mrr/(idx+1))
    logger.info("acc1: {},\t acc5: {},\t acc10: {},\t mrr: {}\t rec_fail: {}".format(\
                            acc1/len(dataset), acc5/len(dataset), acc10/len(dataset), mrr/len(dataset), recall_failed/len(dataset)))
    # logger.info("label_list" + str(label_list))
    # logger.info("pred_list" + str(pred_list))
    return acc1/idx, acc5/idx, acc10/idx, mrr/idx, recall_failed/idx

def pred_model(args, dataset, model):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrain_name)
    # softmax = nn.Softmax(dim=1)
    logger.info("start predicting")
    model.eval()
    loop = tqdm(dataset, total=len(dataset))
    pred_list = []
    for idx, data in enumerate(loop):
        riddle = data[0]
        recall_list = data[1]
        recall_dataset = RecallDataset(recall_list)
        recall_dataloader = Data.DataLoader(dataset=recall_dataset, batch_size=args.batch_size, shuffle=False)
        logit = []
        for recall in recall_dataloader:
            riddle_copy = [riddle for _ in range(len(recall))]
            # pdb.set_trace()
            test_input = tokenizer(text=riddle_copy, text_pair=recall, padding=True, truncation=True, max_length=50, return_tensors='pt')
            if torch.cuda.is_available():
                test_input["input_ids"] = test_input["input_ids"].cuda()
                test_input["token_type_ids"] = test_input["token_type_ids"].cuda()
                test_input["attention_mask"] = test_input["attention_mask"].cuda()
            logit_batch = model(test_input).reshape(-1)
            logit_batch = logit_batch.detach().cpu().numpy().tolist()
            logit += logit_batch
        
        # pdb.set_trace()
        pred = softmax(logit).tolist()
        # acc
        pred_id = [[pred[i], i] for i in range(len(pred))]
        pred_id.sort(reverse=True, key=lambda x: x[0])
        # id_sort = [_[1] for _ in pred_id]
        pred_list.append(pred_id[:10])

    return pred_list


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', default='', type=str, required=True, help='Model Saved Path, Output Directory')
    parser.add_argument("--bert_pretrain_name", default='bert-base-chinese')
    parser.add_argument('--bert_pretrain_path', default='', type=str)
    parser.add_argument('--train_file', default='../data/train_small.csv', type=str)

    parser.add_argument('--model_reload_path', default='', type=str, help='pretrained model to finetune')

    parser.add_argument('--dev_file', default='../data/valid_small2.csv', type=str)
    parser.add_argument('--test_file', default='../data/test_small.txt', type=str)
    parser.add_argument('--gold_file', default='../data/dict.txt', type=str)
    parser.add_argument('--train_recall_file', default='../data/train_recall_small.txt', type=str)
    parser.add_argument('--dev_recall_file', default='../data/valid_recall_small2.txt', type=str)
    parser.add_argument('--test_recall_file', default='../data/test_recall_small.txt', type=str)

    parser.add_argument('--chaizi_file', default='../data/chaizi-jt.txt', type=str)
    parser.add_argument('--neg_rate', default=10, type=int)
    parser.add_argument('--use_riddle_radicle', action='store_true')
    parser.add_argument('--use_recall', action='store_true')
    parser.add_argument('--use_recall_pos', action='store_true')
    parser.add_argument('--use_ans_not_radicle', action='store_true')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--n_val', default=1, type=int, help='conduct validation every $n_val epochs')
    parser.add_argument('--seed', default=42, type=int, help='universal seed')

    parser.add_argument('--only_test', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args)

    if os.path.exists(os.path.join(args.output_dir, "log.txt")) and not args.only_test:
        print("remove log file")
        os.remove(os.path.join(args.output_dir, "log.txt"))
    if args.only_test:
        handler = logging.FileHandler(os.path.join(args.output_dir, "log_test.txt"))
    else:
        handler = logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    for arg in vars(args):
        logger.info('{}: {}'.format(arg, getattr(args, arg)))

    # train_data, val_data
    train_data = BertDataset(args, args.train_file, args.train_recall_file, args.neg_rate)
    val_data = BertTestDataset(args, args.dev_file, args.dev_recall_file)
    pred_data = BertPredDataset(args)

    # initialize model
    model = initial_model(args)

    if torch.cuda.is_available():
        model.cuda()

    if (args.model_reload_path != ""):
        logger.info("loading model " + args.model_reload_path)
        model.load_state_dict(torch.load(args.model_reload_path))

    # train
    if not args.only_test:
        train_model(args, train_data, val_data, model)
    
    # predict
    pred_result = pred_model(args, pred_data, model)
    results = []
    scores = []
    for i in range(len(pred_result)):
        recall_list = pred_data.recall_list[i] if args.use_recall else pred_data.golds_set
        riddle_result = []
        score_result = []
        for j in range(len(pred_result[i])):
            id = pred_result[i][j][1]
            score = pred_result[i][j][0]
            char = recall_list[id]
            riddle_result.append(char)
            score_result.append(score)
        results.append(riddle_result)
        scores.append(score_result)
    with open(os.path.join(args.output_dir, "pred_result.txt"), "w") as f:
        for line in results:
            for char in line:
                f.write(char)
                f.write('\t')
            f.write('\n')

if __name__ == "__main__":
    main()
