import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
from core import DataProcess
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = 'DengXian'
sns.set_palette(sns.hls_palette(s=0.6, n_colors=10))

torch.set_default_dtype(torch.float)
loss = torch.nn.MSELoss()


def train(model, X, Y, num_folds, num_epochs, learning_rate, weight_decay):
    kf = KFold(n_splits=num_folds, shuffle=True)

    # 将特征数据和目标变量转换为 TensorDataset
    dataset = TensorDataset(X, Y)
    train_ls = []
    val_ls = []
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"\rFold {fold + 1}")
        train_data = Subset(dataset, train_index)
        val_data = Subset(dataset, val_index)

        # 创建数据加载器
        batch_size = 64
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        tq = tqdm.tqdm(total=num_epochs * len(train_loader))

        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # 在每个折上训练和评估模型
        # 在这里训练模型，并使用 train_loader 进行训练，val_loader 进行评估
        for epoch in range(num_epochs):
            # 训练模型
            for inputs, targets in train_loader:
                # 运行训练步骤
                l = loss(model(inputs.float()), targets.float())
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                # with torch.no_grad():
                #     # 将⼩于1的值设成1，使得取对数时数值更稳定
                #     clipped_preds = torch.max(net(data), torch.tensor(1.0))
                #     train_ls.append(torch.sqrt(2 * loss(clipped_preds.log(),targets.log()).mean()).item())
                tq.update()
            # 评估模型
            with torch.no_grad():
                for inputs, targets in val_loader:
                    clipped_preds = torch.max(model(inputs), torch.tensor(1.0))
                    val_ls.append(torch.sqrt(2 * loss(clipped_preds.log(), targets.log()).mean()).item())
                    tq.update()
        tq.close()

    # return train_ls, val_ls
    return val_ls


def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


if __name__ == "__main__":
    dp = DataProcess(from_csv=False, from_pt=True)
    data = dp.data
    target = dp.target
    # data = data.to('cuda')
    # target = target.to('cuda')
    net = get_net(data.shape[1])
    net = net.float()
    # net = net.to('cuda')
    loss = train(net, data, target, 5, 100, 0.01, 0.001)
    print(loss)
    loss = pd.DataFrame(loss)
    loss.plot()
    plt.show()
