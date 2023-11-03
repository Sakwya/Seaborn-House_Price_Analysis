import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns

sns.set_style("darkgrid")
sns.set_palette(sns.hls_palette(n_colors=14))
plt.rcParams['font.sans-serif'] = 'DengXian'

df = pd.read_csv("house_info.csv", encoding="utf-8")
district = df['区域'].to_numpy()
D = df['室'].to_numpy()
L = df['厅'].to_numpy()
area = df['房间大小'].to_numpy()
windows = df['窗户朝向'].to_numpy()
fixture = df['装修造型'].to_numpy()
year = df['修建年份'].to_numpy()
hType = df['房源'].to_numpy()
price = df['平米房价'].to_numpy()


da = []
db = []
dc = []
districts = np.unique(district)
# for district_name in districts:
#     bool = [district == district_name]
#     price_ = price[tuple(bool)]
#     da.append(price_)
#     dc.append(pd.DataFrame(price_, columns=[district_name]))
#
# for df in da:
#     db.append(df[:2217])
# da = np.asarray(db)
# df = pd.DataFrame(da.T, columns=districts)
# df.plot.hist(bins=100, alpha=0.5, figsize=(10, 6))
# plt.ioff()
# plt.show()
# fig = plt.figure(figsize=(10, 6))
# gs = gridspec.GridSpec(3, 5)
# axs = []
# for i in range(3):
#     for j in range(5):
#         axs.append(plt.subplot(gs[i, j]))
# for i in range(len(dc)):
#     df = dc[i]
#     df.plot.hist(bins=100, alpha=0.5, figsize=(10, 6), ax=axs[i])
#
# plt.show()

dd = np.asarray([district, fixture, hType, price]).T
df = pd.DataFrame(dd, columns=["区域", "装修造型", "房源", "平米房价"])
sns.stripplot(x="区域", y="平米房价", hue="房源", data=df, jitter=True, size=1)
plt.tight_layout()
plt.show()
sns.violinplot(x="区域", y="平米房价", hue="房源",split=False, data=df,linewidth=0)
plt.tight_layout()
plt.show()
