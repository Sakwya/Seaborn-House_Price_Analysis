import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv('D:/Desktop/不会真的有人学习吧不会吧不会吧/大三上/Secondhand room predict/kaggle_house_price/test.csv')
print(train_data.shape)