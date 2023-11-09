import pandas as pd
import numpy as np

data = pd.read_csv("./house_info_new.csv")

# print(data)
dataD = data['窗户朝向'].str.find("东")
dataN = data['窗户朝向'].str.find("南")
dataX = data['窗户朝向'].str.find("西")
dataB = data['窗户朝向'].str.find("北")
print(dataD, dataN, dataX, dataB)
if dataD > 0:
    dataD = 1
else:
    dataD = 0
# data['修建年份'] = data['修建年份'].str.extract(r'(\d+)').astype(int)

# data.to_csv("./house_info_new.csv", index=False)
