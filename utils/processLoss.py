import os

import pandas as pd

path = r"D:\Documents\Post-Lab\Papers\neuips2025\SEM-main"
logName = "2025-5-9-1-6.txt"
filePath = os.path.join(path, "logs", logName)
savePath = os.path.join(path, "CSVs", logName.split(".")[0]+".csv")

res = []
with open(filePath, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if(i<4):
            continue
        if(i >= 104):
            break
        list0 = line.split(". ")
        newLine = ', '.join(list0)
        list1 = newLine.split(", ")
        print(list1)
        temp = []
        for j, lV in enumerate(list1):
            if(j==0):
                continue
            value = lV.split(":")[1]
            temp.append(float(value))
        res.append(temp)

res = pd.DataFrame(res, columns=['loss1', 'loss2', 'loss3', 'loss4'])
res.to_csv(savePath, index=False)