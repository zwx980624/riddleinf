from distutils.sysconfig import customize_compiler
from http.client import NETWORK_AUTHENTICATION_REQUIRED
import pandas as pd
import numpy as np
import json
import threading
from tqdm import tqdm

train_path = 'data/train.csv'
valid_path = 'data/valid.csv'
cz_path = 'data/chaizi-jt.txt'
filter_path = 'data/常用汉字库 9933.txt'

test_riddle_path = 'data/test.txt'
test_dict_path = 'data/dict.txt'

train_data = pd.read_csv(train_path, header=None)
valid_data = pd.read_csv(valid_path, header=None)
test_riddle = pd.read_csv(test_riddle_path, header=None)
test_dict = pd.read_csv(test_dict_path, header=None)

train_data.columns = ['answer', 'riddle']
valid_data.columns = ['answer', 'riddle']
test_riddle.columns = ['riddle']
test_dict.columns = ['answer']


# txt文件转为csv
def txt2csv(path, topath):
    txt = pd.read_table(path)
    df = pd.DataFrame(txt)
    df.to_csv(topath, index=False)

# txt2csv(test_riddle_path, 'data/test.csv')
# txt2csv(test_dict_path, 'data/dict.csv')

# 提取list中指定index的值   
def ValAccIdx(lst, idxlst):
    i = 0
    ans = []
    while i < len(idxlst):
        ans.append(lst[idxlst[i]])
        i += 1
    return ans

# 创建拆字字典，key：字形， value：拆字后的部首、笔画等
cz_dict = {} 
with open(cz_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()     
    for line in lines:
        line = line.replace('\t',' ').strip('\n').split(' ')
        char = line[0]
        char_cz = line[1:]
        char_cz_set = set(char_cz)
        char_cz = list(char_cz_set)
        cz_dict[char] = char_cz 

# 给定谜语，输出拆字后的部首、笔画等
def riddle_cz(riddle):
    cz_result = []
    for char in riddle:
        if char in cz_dict:
            # cz_result 末尾添加拆字
            cz_result.extend(cz_dict[char])
            # cz_result.extend(word_cz(char))
    return list(set(cz_result))

# # 给定谜语，输出拆字后的部首、笔画等
# def riddle_cz(riddle):
#     cz_result = []
#     for char in riddle:
#         if char in cz_dict:
#             char_cz = cz_dict[char]
#             for x in char_cz:
#                 if x in cz_dict:
#                     cz_result.extend(cz_dict[x])
#                 else:
#                     cz_result.append(x)
#     return list(set(cz_result))

# 输出单个字拆字后的部首、笔画等
def word_cz(word):
    cz_result = []
    if word not in cz_dict:
        return []
    cz = cz_dict[word]
    for x in cz:
        if x in cz_dict:
            cz_result.extend(word_cz(x))
        else:
            cz_result.append(x)
    return list(set(cz_result))

# 将候选答案用常见汉字表进行筛选过滤
def commonword_filter(flt_path, cdd_ans):
    filter_data = []
    with open(flt_path, 'r', encoding='utf-8') as f:
        line = f.readline()  
    filter_data = set(line)
    new_cdd = []
    for x in cdd_ans:
        if x in filter_data:
            new_cdd.append(x)
    return new_cdd

# 将候选答案用所有谜底进行筛选过滤
def answerword_filter(answers, cdd_ans):
    answers = set(answers)
    new_cdd = []
    for x in cdd_ans:
        if x in answers:
            new_cdd.append(x)
    return new_cdd

# def cal_answer_all(riddle, syn_json, data):
#     # 谜语的拆字
#     r_cz = riddle_cz(riddle=riddle)
#     keys = list(cz_dict.keys())
#     cdd_ans = []
#     for i in range(len(keys)):
#         a_cz = cz_dict[keys[i]]
#         # a_cz = riddle_cz(keys[i])
#         num = 0
#         tol = 0.3
#         for x in a_cz:
#             if x in r_cz:
#                 num += 1
#         if num/len(a_cz) >= tol:
#             cdd_ans.append(keys[i])    
#         # if set(a_cz) < set(r_cz) and keys[i] not in riddle:
#         #     cdd_ans.append(keys[i])
#     cdd_ans = commonword_filter(filter_path, cdd_ans)
#     cdd_ans = answerword_filter(data, cdd_ans)
#     return cdd_ans

def cal_answer_all(riddle, syn_ele, data):
    # 谜语的拆字
    r_cz = riddle_cz(riddle=riddle)
    # 谜语同义词的拆字
    s_cz = riddle_cz(riddle=syn_ele["synonym_list"])
    keys = list(cz_dict.keys())
    # keys = list(data)
    cdd_ans = []
    for i in range(len(keys)):
        # a_cz = cz_dict[keys[i]]
        a_cz = riddle_cz(keys[i])
        num = 0
        tol = 0.3
        for x in a_cz:
            if x in r_cz or x in s_cz:
                num += 1 
        if len(a_cz) + len(s_cz) > 0 and num/(len(a_cz)) >= tol and keys[i] not in riddle:
            cdd_ans.append(keys[i])    
        # if set(a_cz) < set(r_cz) and keys[i] not in riddle:
        #     cdd_ans.append(keys[i])
    cdd_ans = commonword_filter(filter_path, cdd_ans)
    cdd_ans = answerword_filter(data, cdd_ans)
    return cdd_ans

# 记录候选谜底的每个部首来源于谜语的index
def cal_cdd_label(riddle, syn_ele, cdd_ans):
    # 谜语的长度
    LenOfRid = len(riddle)
    LenOfSyn = len(syn_ele["synonym_list"])
    ans_cz = cz_dict[cdd_ans]
    # ans_cz = riddle_cz(cdd_ans)
    cdd_cz = []
    cdd_idx = []
    for x in ans_cz:
        for i in range(LenOfRid):
            if riddle[i] in cz_dict and x in riddle_cz(riddle[i]):
                cdd_cz.append(x)
                cdd_idx.append([i, i + 1])
                break
            else:
                for j in range(LenOfSyn):
                    if syn_ele["synonym_list"][j] in cz_dict and x in riddle_cz(syn_ele["synonym_list"][j]):
                        cdd_cz.append(x)
                        cdd_idx.append(syn_ele["pos"][j])
                        break
                              
    return cdd_cz, cdd_idx

# 计算准确率
def cal_accuracy(dataType):
    n_right = 0 # 猜对谜底的数量
    if dataType == 'train':
        riddles = train_data['riddle']
        answers = train_data['answer']
    elif dataType == 'valid':
        riddles = valid_data['riddle']
        answers = valid_data['answer']
    elif dataType == 'test':
        riddles = test_riddle['riddle']
        answers = test_dict['answer']
    else:
        raise ValueError("Input correct data type!")
    length = 0
    LenOfRid = len(riddles)
    for i in range(len(riddles)):
        # 候选谜底
        with open("data/{}_synonym.json".format(dataType), "r") as f:
            syn_json = json.load(f)
        cdd_ans = cal_answer_all(riddles[i], syn_json[i], answers)
        length += len(cdd_ans)
        if answers[i] in cdd_ans:
            n_right += 1
    return n_right/LenOfRid, length/LenOfRid

# 写入json文件
def write_json(dataType):
    if dataType == 'train':
        riddles = train_data['riddle']
        answers = train_data['answer']
    elif dataType == 'valid':
        riddles = valid_data['riddle']
        answers = valid_data['answer']
    elif dataType == 'test':
        riddles = test_riddle['riddle']
        answers = test_dict['answer']
    else:
        raise ValueError("Input correct data type!")
    json_list = []
    # with open("data/{}_synonym.json".format(dataType), "r") as f:
    #     syn_json = json.load(f)
    n_length = 0
    n_right = 0
    LenOfRid = len(riddles)
    for i in tqdm(range(len(riddles))):
        # cdd_ans = cal_answer_all(riddles[i], syn_json[i], answers)
        cdd_ans = cal_answer_all(riddles[i], 0, answers)
        n_length += len(cdd_ans)
        if answers[i] in cdd_ans:
            n_right += 1
        recall_list = []
        for x in cdd_ans:
            # rad, pos = cal_cdd_label(riddles[i], syn_json[i], x)
            rad, pos = cal_cdd_label(riddles[i], 0, x)
            recall_ele = {
                "ans": x, 
                "rad": rad,
                "pos": pos, 
            }
            recall_list.append(recall_ele)     
        json_ele = {
            "id": i, 
            "riddle": riddles[i], 
            "radical": riddle_cz(riddles[i]),
            "recall": recall_list
        }
        json_list.append(json_ele)
    with open("data/{}_data.json".format(dataType), "w", encoding='utf-8') as f:
        json.dump(json_list, f, indent=4, ensure_ascii=False)
    return n_right/LenOfRid, n_length/LenOfRid


def write_txt(dataType):
    if dataType == 'train':
        riddles = train_data['riddle']
        answers = train_data['answer']
    elif dataType == 'valid':
        riddles = valid_data['riddle']
        answers = valid_data['answer']
    elif dataType == 'test':
        riddles = test_riddle['riddle']
        answers = test_dict['answer']
    else:
        raise ValueError("Input correct data type!")
    LenOfRid = len(riddles)
    file_write_obj = open("data/{}_data_3.txt".format(dataType), 'w')
    n_length = 0
    n_right = 0
    with open("data/{}_synonym_3.json".format(dataType), "r") as f:
        syn_json = json.load(f)
    for i in tqdm(range(len(riddles))):
        cdd_ans = cal_answer_all(riddles[i], syn_json[i], answers)
        # cdd_ans = cal_answer_all(riddles[i], 0, answers)
        n_length += len(cdd_ans)
        # if answers[i] in cdd_ans:
        #     n_right += 1   
        file_write_obj.writelines(cdd_ans)
        file_write_obj.write('\n')
    file_write_obj.close()
    return n_right/LenOfRid, n_length/LenOfRid
