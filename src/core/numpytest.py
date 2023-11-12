import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# 格式设置
sns.set_style("darkgrid")
sns.set_palette(sns.hls_palette(n_colors=14))
plt.rcParams['font.sans-serif'] = 'DengXian'

# 读取数据并分解为一维数组
df = pd.read_csv("data/house_info.csv", encoding="utf-8")
np_district = df['区域'].to_numpy()
np_D = df['室'].to_numpy()
np_L = df['厅'].to_numpy()
np_area = df['房间大小'].to_numpy()
np_windows = df['窗户朝向'].to_numpy()
np_fixture = df['装修造型'].to_numpy()
np_year = df['修建年份'].to_numpy()
np_hType = df['房源'].to_numpy()
np_env = df['绿化率'].to_numpy()
np_price = df['平米房价'].to_numpy()

dfA = []  # 源价格列表
dfB = []  # 处理后的价格列表
dfC = []  # 以区域区分的数据集
districts = np.unique(np_district)
for district_name in districts:
    isDistrict = [np_district == district_name]
    price = np_price[tuple(isDistrict)]
    dfA.append(price)
    dfC.append(pd.DataFrame(price, columns=[district_name]))

for df in dfA:
    dfB.append(df[:2217])
dfA = np.asarray(dfB)
# 绘制整体的柱状图
df = pd.DataFrame(dfA.T, columns=districts)
df.plot.hist(bins=100, alpha=0.5, figsize=(10, 6))

# dx = np.asarray([np_price, np_area, np_district, np_env]).T
# dx = pd.DataFrame(dx, columns=["平米价格", "房间大小", "区域", "绿化率"])

# 绘制单个区域的柱状图
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(3, 5)
axs = []
for i in range(3):
    for j in range(5):
        axs.append(plt.subplot(gs[i, j]))
for i in range(len(dfC)):
    df = dfC[i]
    sns.histplot(df, kde=True, ax=axs[i])

fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
axa = plt.subplot(gs[0, 0])
axb = plt.subplot(gs[1, 0])
dd = np.asarray([np_district, np_fixture, np_hType, np_price]).T
df = pd.DataFrame(dd, columns=["区域", "装修造型", "房源", "平米房价"])
sns.stripplot(x="区域", y="平米房价", hue="房源", data=df, jitter=True, size=2, ax=axa)
sns.violinplot(x="区域", y="平米房价", hue="房源", split=False, data=df, linewidth=0, ax=axb)
plt.tight_layout()
plt.show()
