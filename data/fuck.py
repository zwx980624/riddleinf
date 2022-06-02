import csv
import random

data = []
with open('valid.csv')as f:
    #f_csv = csv.reader(f)
    for row in f:
        data.append(row)
random.shuffle(data)
with open("valid_smal.csv", "w") as f:
    for i in range(2000):
        f.write(data[i])

