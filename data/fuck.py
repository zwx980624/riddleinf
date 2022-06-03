import csv
import random

# data = []
# with open('valid.csv')as f:
#     #f_csv = csv.reader(f)
#     for row in f:
#         data.append(row)
# random.shuffle(data)
# with open("valid_smal.csv", "w") as f:
#     for i in range(2000):
#         f.write(data[i])

fvalid =  open('valid.csv')
fvalid_small = open('valid_small.csv')
data = []
for row in fvalid:
    data.append(row)
data_small = []
for row in fvalid_small:
    data_small.append(row)

frecall =  open('valid_recall.txt')
lines = frecall.readlines()
data_recall = []
for line in lines:
    data_recall.append(line)

with open("valild_recall_small.txt", "w") as f:
    for small in data_small:
        i = data.index(small)
        print(i)
        f.write(data_recall[i])