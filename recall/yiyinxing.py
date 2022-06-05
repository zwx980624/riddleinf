import jieba
import jieba.analyse
import pandas as pd
import json
from collections import Counter
from kmcha import kmcha_same
from kmcha import kmcha_similar
from similar import pattern_similar
from tqdm import tqdm
from pypinyin import pinyin, Style
import pypinyin


def load_pinyin_dict(file_path):
    pinyin_dict = {} 
    for line in open(file_path,'r',encoding='utf-8'):
        line = line.strip('\n')
        line = line.split('\t')
        pinyin_dict[line[0]] = line[1:]
    return pinyin_dict


def load_dict(file_path):
    combine_dict = {}
    for line in open(file_path,'r',encoding='utf-8'):
        seperate_word = line.strip().split(",")
        del seperate_word[1]
        if seperate_word[0] not in combine_dict.keys():
            combine_dict[seperate_word[0]] = [seperate_word[1]]
        else:
            combine_dict[seperate_word[0]].append(seperate_word[1])
    return combine_dict
    
def load_data(file_path):
    data = pd.read_csv(file_path,header=None)
    riddles = data[1]
    return riddles

def load_test(file_path):
    xs = []
    for line in open(file_path,'r',encoding='utf-8'):
        x = line.rstrip('\n')
        xs.append(x)
    return xs

def tone_simi_func(char,pinyin_dict):
    py = pinyin(char, style=Style.NORMAL)[0][0]
    return pinyin_dict[py][:10]

def replacesynonym(strings,combine_dict,pinyin_dict):
    seg_list = jieba.cut(strings, cut_all=False)
    f = "/".join(seg_list).encode("utf-8")
    f = f.decode("utf-8")
    #seg_list = jieba.analyse.extract_tags(strings, topK=20, withWeight=False, allowPOS=('n','ns','nz'))
    #seg_list
    synonym = []
    pos_list = []
    char_shape = []
    char_shape_pos = []
    for word in f.split('/'):
        #word是一个词语
        #先遍历每一个字，得到十个形近字和十个同音字
        
        for char in word:
            #形近字
            char_simi = pattern_similar(char)
            char_shape.extend(char_simi)
            char_shape_pos.extend([strings.index(char)]*len(char_simi))
            
            #同音字
            try:
                tone_simi = tone_simi_func(char,pinyin_dict)
                char_shape.extend(tone_simi)
                char_shape_pos.extend([strings.index(char)]*len(tone_simi))
            except:
                pass

        if word in combine_dict:
            pos = strings.find(word)
            pos = [pos,pos+len(word)]
            # word_kmcha = kmcha_same(word) + kmcha_similar(word)
            #这里的word还是词语 不能直接用pattern_similar
            word = combine_dict[word]
            #word是一个词的列表
            # word = word + word_kmcha
            synonym.append(word)
            pos_list.append(pos)
            #转换的时候同步给出位置信息
            #synonym和pos_list的位置信息是对应的
        else:
            pass
            # final_sentence.append(word)
    for i in range(len(char_shape_pos)):
        char_shape_pos[i] = [char_shape_pos[i],char_shape_pos[i]+1]
    
        
            
    word_list = []
    pos_li = []
    for word_li in synonym:
        index = synonym.index(word_li)
        po = pos_list[index]
        #word的位置是
        #index = synonym.index(word_li)
        #pos_list[index]
        for word in word_li:
            for j in word:
                word_list.append(j)
                pos_li.append(po)
                #得到了j，还需要j所对应的word的位置    
    word_set = []
    list_set = []         
    word_se = Counter(word_list).most_common(10)
    for i in word_se:
        word_set.append(i[0])
        index = word_list.index(i[0])
        list_set.append(pos_li[index])
    word_set.extend(char_shape)
    list_set.extend(char_shape_pos)
    return word_set,list_set

def main(file_path,dict_path,pinyin_dict_path):
    riddles = load_data(file_path)
    combine_dict = load_dict(dict_path)
    pinyin_dict = load_pinyin_dict(pinyin_dict_path)
    outputs = []
    for i in tqdm(range(len(riddles))):
        word_set,list_set = replacesynonym(riddles[i],combine_dict,pinyin_dict)
        #构造一个字典，输出json
        output = {
            "id":i,
            "riddle":riddles[i],
            "synonym_list":word_set,
            "pos":list_set,
        }
        outputs.append(output)
    return outputs




if __name__ == '__main__':
    ouput_train = main('./data/train.csv','./data/同义关系库.txt','./data/chinese_homophone_char.txt')
    ouput_valid = main('./data/valid.csv','./data/同义关系库.txt','./data/chinese_homophone_char.txt')
    with open('./data/train_synonym_3.json','w',encoding='utf-8') as f:
        json.dump(ouput_train,f,ensure_ascii=False)
    print('train_gene_done!')
    with open('./data/valid_synonym_3.json','w',encoding='utf-8') as f:
        json.dump(ouput_valid,f,ensure_ascii=False)
    print('valid_gene_done!')

    riddles = load_test('./data/test.txt')
    combine_dict = load_dict('./data/同义关系库.txt')
    pinyin_dict = load_pinyin_dict('./data/chinese_homophone_char.txt')
    outputs = []
    for i in tqdm(range(len(riddles))):
        word_set,list_set = replacesynonym(riddles[i],combine_dict,pinyin_dict)
        #构造一个字典，输出json
        output = {
            "id":i,
            "riddle":riddles[i],
            "synonym_list":word_set,
            "pos":list_set,
        }
        outputs.append(output)
    with open('./data/test_synonym_3.json','w',encoding='utf-8') as f:
        json.dump(outputs,f,ensure_ascii=False)
    print("test done!")

    # strs = '先生来到才鞠躬'
    # combine_dict = load_dict('同义关系库.txt')
    # pinyin_dict = load_pinyin_dict('chinese_homophone_char.txt')
    # seg_list = jieba.cut(strs, cut_all=False)
    # f = "/".join(seg_list).encode("utf-8")
    # f = f.decode("utf-8")
    # for word in f.split('/'):
    #     #word是一个词语
    #     #先遍历每一个字，得到十个形近字和十个同音字
        
    #     for char in word:
    #         #形近字
    #         char_simi = pattern_similar(char)
    #         print(char_simi)
    #         #同音字
    #         try:
    #             tone_simi = tone_simi_func(char,pinyin_dict)
    #             print(tone_simi)
    #         except:
    #             pass

    
    
    
    # strs = ['先生来到才鞠躬','卧龙姓氏却昭著，玄德造庐虔顾三']
    # combine_dict = load_dict('同义关系库.txt')
    # pinyin_dict = load_pinyin_dict('chinese_homophone_char.txt')
    # for i in range(len(strs)):
    #     a,b = replacesynonym(strs[i],combine_dict,pinyin_dict)
    #     print(a,b)
        
    
    

