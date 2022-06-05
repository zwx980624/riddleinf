import json

def initDict(path):
   dict = {}; 
   with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            # 移除换行符，并且根据空格拆分
            splits = line.strip('\n').split(' ');
            key = splits[0];
            value = splits[1];
            dict[key] = value; 
   return dict;
   
# 字典初始化 
bihuashuDict = initDict('./db/bihuashu_2w.txt');
hanzijiegouDict = initDict('./db/hanzijiegou_2w.txt');
pianpangbushouDict = initDict('./db/pianpangbushou_2w.txt');
sijiaobianmaDict = initDict('./db/sijiaobianma_2w.txt');

with open('gold_chaizi_set_train.json','r',encoding='utf-8') as f1:
    train_data = json.load(f1)
with open('gold_chaizi_set.json','r',encoding='utf-8') as f2:
    valid_data = json.load(f2)
with open('gold_chaizi_set_test.json','r',encoding='utf-8') as f3:
    test_data = json.load(f3)
word_set = list(set(train_data+valid_data+test_data))

# 权重定义（可自行调整）
hanzijiegouRate = 10;
sijiaobianmaRate = 8;
pianpangbushouRate = 6;
bihuashuRate = 2;

# 计算核心方法
'''
desc: 笔画数相似度
'''
def bihuashuSimilar(charOne, charTwo): 
    valueOne = bihuashuDict[charOne];
    valueTwo = bihuashuDict[charTwo];
    
    numOne = int(valueOne);
    numTwo = int(valueTwo);
    
    diffVal = 1 - abs((numOne - numTwo) / max(numOne, numTwo));
    return bihuashuRate * diffVal * 1.0;

    
'''
desc: 汉字结构数相似度
'''
def hanzijiegouSimilar(charOne, charTwo): 
    valueOne = hanzijiegouDict[charOne];
    valueTwo = hanzijiegouDict[charTwo];
    
    if valueOne == valueTwo:
        # 后续可以优化为相近的结构
        return hanzijiegouRate * 1;
    return 0;
    
'''
desc: 四角编码相似度
'''
def sijiaobianmaSimilar(charOne, charTwo): 
    valueOne = sijiaobianmaDict[charOne];
    valueTwo = sijiaobianmaDict[charTwo];
    
    totalScore = 0.0;
    minLen = min(len(valueOne), len(valueTwo));
    
    for i in range(minLen):
        if valueOne[i] == valueTwo[i]:
            totalScore += 1.0;
    
    totalScore = totalScore / minLen * 1.0;
    return totalScore * sijiaobianmaRate;

'''
desc: 偏旁部首相似度
'''
def pianpangbushoutSimilar(charOne, charTwo): 
    valueOne = pianpangbushouDict[charOne];
    valueTwo = pianpangbushouDict[charTwo];
    
    if valueOne == valueTwo:
        # 后续可以优化为字的拆分
        return pianpangbushouRate * 1;
    return 0;  
    
'''
desc: 计算两个汉字的相似度
'''
def similar(charOne, charTwo):
    if charOne == charTwo:
        return 1.0;
    
    sijiaoScore = sijiaobianmaSimilar(charOne, charTwo);    
    jiegouScore = hanzijiegouSimilar(charOne, charTwo);
    bushouScore = pianpangbushoutSimilar(charOne, charTwo);
    bihuashuScore = bihuashuSimilar(charOne, charTwo);
    
    totalScore = sijiaoScore + jiegouScore + bushouScore + bihuashuScore;    
    totalRate = hanzijiegouRate + sijiaobianmaRate + pianpangbushouRate + bihuashuRate;
    
    
    result = totalScore*1.0 / totalRate * 1.0;
    # print('总分：' + str(totalScore) + ', 总权重: ' + str(totalRate) +', 结果:' + str(result));
    # print('四角编码：' + str(sijiaoScore));
    # print('汉字结构：' + str(jiegouScore));
    # print('偏旁部首：' + str(bushouScore));
    # print('笔画数：' + str(bihuashuScore));
    return result;

def pattern_similar(word):
    #遍历字典，找出与给定的字字形最相近的十个字
    #对word 首先要排除掉标点符号
    scores = [0]*10
    candidata = ['']*10
    for char in word_set:
        try:
            score = similar(word,char)
            if score > min(scores):
                min_index = scores.index((min(scores)))
                scores[min_index] = score
                candidata[min_index] = char
        except:
            pass
    return candidata

'''
$ python main.py
总分：25.428571428571427, 总权重: 26, 结果:0.978021978021978
四角编码：8.0
汉字结构：10
偏旁部首：6
笔画数：1.4285714285714286
'''
if __name__ == "__main__":
    a = pattern_similar('才')
    print(a)
